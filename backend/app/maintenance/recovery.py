"""Verified local-storage snapshots for moving Shuttl between short GPU rentals.

Backup/restore use only explicitly supplied DSNs. Restore accepts an empty database
and a new directory; it never drops an existing database or overwrites a library.
"""
from __future__ import annotations

import argparse
from contextlib import ExitStack, contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timezone
import fcntl
import hashlib
import io
import json
import os
from pathlib import Path, PurePosixPath
import re
import stat
import subprocess
import tarfile
import tempfile
import uuid

from dotenv import dotenv_values
import psycopg2
from psycopg2 import sql
from psycopg2.extras import RealDictCursor, register_uuid
from psycopg2.extensions import parse_dsn

register_uuid()
TABLES = {'users', 'matches', 'rallies', 'shot_points', 'analysis_jobs', 'media_deletions'}
PATH_SETTINGS = ('SHUTTL_WEIGHTS_DIR', 'SHUTTL_HITNET_WEIGHTS', 'SHUTTL_BST_MANIFEST', 'SHUTTL_RTMPOSE_WEIGHTS')
FORMAT = 1


class RecoveryError(RuntimeError):
    pass


def safe_name(name):
    path = PurePosixPath(name)
    if not name or '\\' in name or path.is_absolute() or any(p in {'', '.', '..'} for p in name.split('/')):
        raise RecoveryError('Unsafe archive path')
    return path


def json_bytes(value):
    return (json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True, default=str) + '\n').encode()


def atomic_write(path, data):
    path = Path(path)
    fd, temporary = tempfile.mkstemp(prefix='.' + path.name + '.', dir=path.parent)
    try:
        with os.fdopen(fd, 'wb') as stream:
            stream.write(data); stream.flush(); os.fsync(stream.fileno())
        os.replace(temporary, path)
        directory = os.open(path.parent, os.O_RDONLY)
        try: os.fsync(directory)
        finally: os.close(directory)
    finally:
        Path(temporary).unlink(missing_ok=True)


@contextmanager
def repository_lock(repository):
    repository = Path(repository)
    repository.mkdir(parents=True, exist_ok=True, mode=0o700)
    if repository.is_symlink(): raise RecoveryError('Repository must not be a symlink')
    with (repository / '.recovery.lock').open('a+b') as lock:
        os.chmod(lock.name, 0o600)
        try: fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError: raise RecoveryError('Another recovery operation is using this repository') from None
        yield


@dataclass
class Database:
    dsn: str = field(repr=False)
    container: str | None = None

    def __post_init__(self):
        self.dsn = self.dsn.replace('postgresql+asyncpg://','postgresql://',1).replace('postgres://','postgresql://',1)
        if not self.dsn.startswith('postgresql://'): raise RecoveryError('Use a PostgreSQL connection URL')

    def connect(self):
        return psycopg2.connect(self.dsn, cursor_factory=RealDictCursor, connect_timeout=10,
                               application_name='shuttl-recovery', options='-c lock_timeout=5000')

    def tool(self, program, arguments, *, output=None, input_file=None):
        params = parse_dsn(self.dsn)
        mapping = {'host':'PGHOST', 'port':'PGPORT', 'dbname':'PGDATABASE', 'user':'PGUSER',
                   'password':'PGPASSWORD', 'sslmode':'PGSSLMODE', 'sslcert':'PGSSLCERT',
                   'sslkey':'PGSSLKEY', 'sslrootcert':'PGSSLROOTCERT', 'hostaddr':'PGHOSTADDR',
                   'service':'PGSERVICE'}
        env = {k:v for k,v in os.environ.items() if not k.startswith('PG')}
        keys = []
        for key, var in mapping.items():
            if key in params: env[var] = params[key]; keys.append(var)
        env['PGCONNECT_TIMEOUT'] = '10'; keys.append('PGCONNECT_TIMEOUT')
        command = [program]
        if self.container:
            # The named container must be this PostgreSQL server (standard port 5432).
            env['PGHOST'] = '127.0.0.1'; env['PGPORT'] = '5432'
            keys = sorted(set(keys + ['PGHOST','PGPORT']) - {'PGHOSTADDR'})
            command = ['docker','exec','-i']
            for key in keys: command += ['--env', key]  # values come from env, never argv/logs
            command += [self.container, program]
        result = subprocess.run(command + arguments, stdin=input_file, stdout=output or subprocess.PIPE,
                                stderr=subprocess.PIPE, env=env)
        if result.returncode:
            # libpq's full diagnostics may contain deployment details; keep the CLI summary bounded.
            raise RecoveryError(f'{program} failed with exit code {result.returncode}; check PostgreSQL version, connectivity and permissions')
        return result.stdout


def lineage(conn, *, create=False):
    with conn.cursor() as cur:
        if create:
            cur.execute('''CREATE TABLE IF NOT EXISTS shuttl_recovery_identity (
                singleton boolean PRIMARY KEY DEFAULT true CHECK (singleton), lineage uuid NOT NULL)''')
            cur.execute('INSERT INTO shuttl_recovery_identity(singleton,lineage) VALUES (true,%s) ON CONFLICT DO NOTHING', (uuid.uuid4(),))
        cur.execute('SELECT lineage FROM shuttl_recovery_identity WHERE singleton')
        row = cur.fetchone()
        if not row: raise RecoveryError('Missing database recovery identity')
        return str(row['lineage'])


def table_names(conn):
    with conn.cursor() as cur:
        cur.execute("SELECT tablename FROM pg_tables WHERE schemaname='public'")
        return {r['tablename'] for r in cur.fetchall()}


def worker_lock_key(match_id):
    return int.from_bytes(hashlib.blake2b(('shuttl-analysis:'+str(match_id)).encode(), digest_size=8).digest(), 'big', signed=True)


@contextmanager
def frozen_database(database):
    conn = database.connect()
    try:
        if not TABLES <= table_names(conn): raise RecoveryError('Database is missing Shuttl tables; upgrade schema first')
        identity = lineage(conn, create=True)
        names = sorted(table_names(conn))
        conn.commit()
        conn.set_session(isolation_level='REPEATABLE READ')
        with conn.cursor() as cur:
            # Take the lock BEFORE the first MVCC read/export; otherwise a just-deleted match
            # could be present in the snapshot while its tombstone was committed after that read.
            # EXCLUSIVE still allows pg_dump reads but stops upload publication (FOR UPDATE).
            cur.execute(sql.SQL('LOCK TABLE {} IN EXCLUSIVE MODE').format(
                sql.SQL(',').join(sql.Identifier('public', name) for name in names)))
            cur.execute("SELECT DISTINCT match_id FROM analysis_jobs WHERE state IN ('queued','running','retry')")
            for row in cur.fetchall():
                cur.execute('SELECT pg_try_advisory_xact_lock(%s) AS ok', (worker_lock_key(row['match_id']),))
                if not cur.fetchone()['ok']: raise RecoveryError('An analysis worker is active; stop it before backup')
            cur.execute('SELECT pg_export_snapshot() AS snapshot')
            snapshot = cur.fetchone()['snapshot']
            cur.execute('SELECT * FROM media_deletions ORDER BY match_id')
            deleted = [dict(row) for row in cur.fetchall()]
            cur.execute('SELECT id,status,video_oss_key FROM matches ORDER BY id')
            matches = [dict(row) for row in cur.fetchall()]
            yield conn, identity, snapshot, deleted, matches
    finally:
        conn.rollback(); conn.close()


def file_inventory(root, *, media=False):
    root = Path(root)
    if root.is_symlink() or not root.is_dir(): raise RecoveryError('Source must be a real directory')
    found = {}
    for directory, dirs, files in os.walk(root, followlinks=False):
        relative = Path(directory).relative_to(root)
        if media and relative == Path('.'):
            dirs[:] = [d for d in dirs if d != '.analysis']
        for name in dirs + files:
            path = Path(directory) / name
            if path.is_symlink(): raise RecoveryError('Symbolic links are not supported in recovery sources')
        for name in files:
            if media and name.startswith('.') and name.endswith('.part'): continue
            path = Path(directory) / name
            info = path.stat()
            if not stat.S_ISREG(info.st_mode): raise RecoveryError('Special files are not supported')
            key = path.relative_to(root).as_posix(); safe_name(key)
            found[key] = (info.st_size, info.st_mtime_ns, info.st_ino, info.st_mode & 0o111)
    return found


class HashReader:
    def __init__(self, stream): self.stream=stream; self.digest=hashlib.sha256(); self.count=0
    def read(self, size=-1):
        data=self.stream.read(size); self.digest.update(data); self.count+=len(data); return data


def add_file(archive, manifest, name, source):
    safe_name(name)
    before = source.stat()
    if source.is_symlink() or not stat.S_ISREG(before.st_mode): raise RecoveryError('Source changed type')
    item = tarfile.TarInfo(name); item.size = before.st_size
    item.mode = 0o700 if before.st_mode & 0o111 else 0o600
    with source.open('rb') as stream:
        reader = HashReader(stream); archive.addfile(item, reader)
    after = source.stat()
    if (before.st_size,before.st_mtime_ns,before.st_ino) != (after.st_size,after.st_mtime_ns,after.st_ino):
        raise RecoveryError('Source changed during backup; stop writers and retry')
    manifest['files'][name] = {'size':reader.count,'sha256':reader.digest.hexdigest(),'mode':item.mode}


def read_ledger(repository):
    path = repository / 'latest-deletions.json'
    if not path.exists(): return None
    try:
        ledger=json.loads(path.read_text())
        if ledger['format']!=FORMAT or not isinstance(ledger['revision'],int) or ledger['revision']<1:
            raise ValueError('Unsupported ledger')
        uuid.UUID(ledger['lineage'])
        if not isinstance(ledger['deletions'],list): raise ValueError('Invalid deletions')
        identities=set()
        for row in ledger['deletions']:
            mid=uuid.UUID(row['match_id']);uuid.UUID(row['user_id']);safe_name(row['video_key'])
            datetime.fromisoformat(row['created_at'])
            if mid in identities or row['storage_backend'] not in {'local','oss'}: raise ValueError('Invalid deletion')
            identities.add(mid)
        return ledger
    except (ValueError,TypeError,KeyError) as exc:
        raise RecoveryError('Deletion ledger is damaged or unsupported; preserve it for recovery') from exc


def merged_ledger(previous, identity, deleted):
    if previous is not None and previous['lineage'] != identity: raise RecoveryError('Repository belongs to a different database lineage')
    entries = {r['match_id']:r for r in previous['deletions']} if previous else {}
    for raw in deleted:
        row = json.loads(json_bytes(raw)); prior = entries.get(row['match_id'])
        if prior and any(prior[k] != row[k] for k in ('user_id','video_key','storage_backend')): raise RecoveryError('Deletion identity conflict')
        entries[row['match_id']] = row
    return {'format':FORMAT, 'lineage':identity, 'revision':(previous['revision'] if previous else 0)+1,
            'captured_at':datetime.now(timezone.utc).isoformat(), 'deletions':list(entries.values())}


def backup(database, repository, media, config, assets):
    for source in [Path(repository),Path(media),Path(config),*(Path(p) for p in assets.values())]:
        if source.is_symlink(): raise RecoveryError('Recovery paths must not be symlinks')
    repository=Path(repository).resolve(); media=Path(media).resolve(); config=Path(config).resolve()
    config_before=config.stat()
    values=dotenv_values(config,interpolate=False)
    if values.get('STORAGE_BACKEND') != 'local': raise RecoveryError('Self-contained backup currently requires explicit STORAGE_BACKEND=local')
    if not assets: raise RecoveryError('Include deployed models, source/runtime and training runs with --asset')
    sources={'media':media}
    for name, path in assets.items():
        if not re.fullmatch(r'[a-zA-Z0-9_-]+',name): raise RecoveryError('Invalid asset root name')
        sources['assets/'+name]=Path(path).resolve()
    for path in sources.values():
        if repository.is_relative_to(path) or path.is_relative_to(repository):
            raise RecoveryError('Backup repository and source trees must be separate')
    roots=list(sources.values())
    for index,path in enumerate(roots):
        if any(path.is_relative_to(other) or other.is_relative_to(path) for other in roots[index+1:]):
            raise RecoveryError('Media and asset source roots must not overlap')
    bindings={}
    for key in PATH_SETTINGS:
        if not values.get(key): continue
        path=Path(values[key]).expanduser()
        if not path.is_absolute(): path=config.parent/path
        path=path.resolve()
        if not path.exists(): raise RecoveryError(f'Configured model path is missing: {key}')
        candidates=[(prefix,source) for prefix,source in sources.items() if prefix.startswith('assets/') and path.is_relative_to(source)]
        if not candidates: raise RecoveryError(f'Configured model is outside all archived assets: {key}')
        prefix,source=max(candidates,key=lambda item:len(item[1].parts))
        bindings[key]=(Path(prefix)/path.relative_to(source)).as_posix()
    with repository_lock(repository), ExitStack() as stack:
        # BST's writer uses an exclusive lock. Snapshot keeps it from starting mid-copy.
        for prefix, source in sources.items():
            if prefix == 'media': continue
            for lock_path in source.rglob('.run.lock'):
                if lock_path.is_symlink(): raise RecoveryError('Training lock is a symlink')
                lock=stack.enter_context(lock_path.open('rb'))
                try: fcntl.flock(lock,fcntl.LOCK_SH | fcntl.LOCK_NB)
                except BlockingIOError: raise RecoveryError('Training is active; save and stop it before backup') from None
        with frozen_database(database) as (conn, identity, snapshot, deleted, matches):
            ledger=merged_ledger(read_ledger(repository),identity,deleted)
            snapshot_id=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')+'-'+uuid.uuid4().hex[:12]
            manifest={'format':FORMAT,'snapshot_id':snapshot_id,'lineage':identity,'revision':ledger['revision'],
                      'created_at':ledger['captured_at'],'storage':'local','files':{},'path_bindings':bindings,
                      'sources':{k:str(v) for k,v in sources.items()},'matches':matches}
            before={prefix:file_inventory(source,media=prefix=='media') for prefix,source in sources.items()}
            for row in matches:
                key=row['video_oss_key']
                if key:
                    safe_name(key)
                    if key not in before['media'] and row['status'] != 'uploading':
                        raise RecoveryError('A non-uploading match is missing its source video')
            with tempfile.TemporaryDirectory(prefix='.backup-',dir=repository) as temporary:
                stage=Path(temporary); dump=stage/'database.dump'
                with dump.open('wb') as out:
                    database.tool('pg_dump',['--format=custom','--no-owner','--no-acl','--snapshot='+snapshot],output=out)
                archive_path=stage/(snapshot_id+'.tar')
                with tarfile.open(archive_path,'w',format=tarfile.PAX_FORMAT) as archive:
                    add_file(archive,manifest,'database.dump',dump)
                    add_file(archive,manifest,'config/source.env',config)
                    for prefix,source in sources.items():
                        for relative in sorted(before[prefix]):
                            add_file(archive,manifest,prefix+'/'+relative,source/relative)
                    for prefix,source in sources.items():
                        if before[prefix] != file_inventory(source,media=prefix=='media'):
                            raise RecoveryError('Source tree changed during backup; stop writers and retry')
                    config_after=config.stat()
                    if (config_before.st_size,config_before.st_mtime_ns,config_before.st_ino)!=(config_after.st_size,config_after.st_mtime_ns,config_after.st_ino):
                        raise RecoveryError('Configuration changed during backup')
                    data=json_bytes(manifest); item=tarfile.TarInfo('manifest.json');item.size=len(data);item.mode=0o600
                    archive.addfile(item,io.BytesIO(data))
                os.chmod(archive_path,0o600)
                with archive_path.open('rb') as stream: os.fsync(stream.fileno())
                verify(archive_path)
                # Publish deletion high-water mark first. A failed archive publish cannot forget a deletion.
                atomic_write(repository/'latest-deletions.json',json_bytes(ledger))
                final=repository/archive_path.name
                os.link(archive_path,final)
                directory=os.open(repository,os.O_RDONLY)
                try: os.fsync(directory)
                finally: os.close(directory)
                return {'snapshot':str(final),'files':len(manifest['files']),'revision':ledger['revision'],'lineage':identity}


def verify(archive_path, *, extract_to=None):
    seen={}; manifest=None
    with tarfile.open(archive_path,'r:') as archive:
        for item in archive:
            safe_name(item.name)
            if not item.isfile() or item.name in seen: raise RecoveryError('Archive contains links, special files or duplicates')
            if item.mode not in {0o600,0o700}: raise RecoveryError('Unsupported file permissions')
            stream=archive.extractfile(item)
            if item.name=='manifest.json':
                if manifest is not None or item.size>64*1024**2: raise RecoveryError('Invalid manifest')
                manifest=json.load(stream);seen[item.name]=None;continue
            digest=hashlib.sha256();size=0
            target=None
            if extract_to is not None:
                target=Path(extract_to)/item.name
                target.parent.mkdir(parents=True,exist_ok=True,mode=0o700)
            with ExitStack() as stack:
                output=stack.enter_context(target.open('xb')) if target else None
                for chunk in iter(lambda:stream.read(1<<20),b''):
                    digest.update(chunk);size+=len(chunk)
                    if output: output.write(chunk)
                if output:
                    output.flush();os.fsync(output.fileno());os.chmod(target,item.mode)
            seen[item.name]={'size':size,'sha256':digest.hexdigest(),'mode':item.mode}
    if not manifest or manifest.get('format')!=FORMAT: raise RecoveryError('Unsupported or missing manifest')
    seen.pop('manifest.json',None)
    if seen!=manifest['files']: raise RecoveryError('Backup inventory/checksum mismatch')
    if not {'database.dump','config/source.env'}<=seen.keys(): raise RecoveryError('Incomplete backup')
    uuid.UUID(manifest['lineage'])
    for key in manifest['path_bindings'].values(): safe_name(key)
    return manifest


def replay_deletions(conn, ledger):
    with conn.cursor() as cur:
        for row in ledger['deletions']:
            mid=uuid.UUID(row['match_id']);owner=uuid.UUID(row['user_id']);safe_name(row['video_key'])
            cur.execute('SELECT user_id,video_oss_key FROM matches WHERE id=%s',(mid,))
            match=cur.fetchone()
            if match and (match['user_id']!=owner or match['video_oss_key'] not in {None,row['video_key']}): raise RecoveryError('Deletion conflicts with restored match identity')
            cur.execute('SELECT user_id,video_key,storage_backend FROM media_deletions WHERE match_id=%s',(mid,))
            prior=cur.fetchone()
            if prior and (prior['user_id']!=owner or any(prior[k]!=row[k] for k in ('video_key','storage_backend'))): raise RecoveryError('Conflicting deletion ledger')
            cur.execute('''INSERT INTO media_deletions(match_id,user_id,video_key,storage_backend,state,attempts,created_at,available_at)
                VALUES (%s,%s,%s,%s,'pending',0,%s,now()) ON CONFLICT(match_id) DO UPDATE SET
                state='pending',available_at=now(),cleaned_at=NULL,last_error=NULL''',
                (mid,owner,row['video_key'],row['storage_backend'],row['created_at']))
            cur.execute('DELETE FROM matches WHERE id=%s',(mid,))
        # Broker contents are deliberately not restored. Database outbox delivers fresh messages.
        # Preserve attempt counters: moving machines must not reset the bounded retry budget.
        cur.execute('''UPDATE analysis_jobs SET state='queued',claim_token=NULL,percent=0,stage='restored',
            dispatched_at=NULL,heartbeat_at=NULL,available_at=now(),error=NULL
            WHERE state IN ('queued','running','retry')''')
        return cur.rowcount


def quote_env(value):
    return '"'+str(value).replace('\\','\\\\').replace('"','\\"').replace('\n','\\n').replace('\r','\\r')+'"'


def restore(database, repository, archive_path, destination, *, redis_url):
    repository=Path(repository).resolve();archive_path=Path(archive_path).resolve();destination=Path(destination).absolute()
    if destination.exists(): raise RecoveryError('Restore destination must not exist')
    if not archive_path.is_relative_to(repository): raise RecoveryError('Snapshot must belong to the selected repository')
    with repository_lock(repository):
        manifest=verify(archive_path)
        ledger=read_ledger(repository)
        if not ledger or ledger['lineage']!=manifest['lineage'] or ledger['revision']<manifest['revision']:
            raise RecoveryError('Latest deletion ledger is missing, stale or from another database')
        conn=database.connect()
        try:
            with conn.cursor() as cur:
                cur.execute("SELECT pg_try_advisory_lock(hashtextextended('shuttl-restore',0)) AS ok")
                if not cur.fetchone()['ok']: raise RecoveryError('Another restore is using this database')
                cur.execute("SELECT 1 FROM pg_class c JOIN pg_namespace n ON n.oid=c.relnamespace WHERE n.nspname NOT IN ('pg_catalog','information_schema') AND n.nspname NOT LIKE 'pg_toast%' AND n.nspname NOT LIKE 'pg_temp%' LIMIT 1")
                if cur.fetchone(): raise RecoveryError('Restore database is not empty; existing data will not be overwritten')
                cur.execute("SELECT 1 FROM pg_type t JOIN pg_namespace n ON n.oid=t.typnamespace WHERE n.nspname='public' UNION ALL SELECT 1 FROM pg_proc p JOIN pg_namespace n ON n.oid=p.pronamespace WHERE n.nspname='public' LIMIT 1")
                if cur.fetchone(): raise RecoveryError('Restore database contains custom types/functions; use a pristine database')
            conn.commit()
            destination.mkdir(parents=True,exist_ok=False,mode=0o700)
            marker=destination/'RESTORE_IN_PROGRESS.json'
            atomic_write(marker,json_bytes({'state':'in_progress','snapshot':manifest['snapshot_id']}))
            stage=destination/'.staging';stage.mkdir(mode=0o700)
            try:
                if verify(archive_path,extract_to=stage)!=manifest:
                    raise RecoveryError('Snapshot changed between verification and extraction')
                with (stage/'database.dump').open('rb') as stream:
                    database.tool('pg_restore',['--single-transaction','--exit-on-error','--no-owner','--no-acl','--dbname='+parse_dsn(database.dsn)['dbname']],input_file=stream)
                with conn:
                    if lineage(conn)!=manifest['lineage']: raise RecoveryError('Restored database lineage mismatch')
                    pending=replay_deletions(conn,ledger)
                # Delete restored local media using the merged high-water mark, before publishing runtime.env.
                for row in ledger['deletions']:
                    (stage/'media'/str(safe_name(row['video_key']))).unlink(missing_ok=True)
                (stage/'media').mkdir(exist_ok=True,mode=0o700)
                for name in ('media','assets','config'):
                    source=stage/name
                    if source.exists(): source.rename(destination/name)
                values=dotenv_values(destination/'config/source.env',interpolate=False)
                values['DATABASE_URL']=database.dsn.replace('postgresql://','postgresql+asyncpg://',1)
                values['REDIS_URL']=redis_url;values['MEDIA_DIR']=str(destination/'media')
                for key, relative in manifest['path_bindings'].items(): values[key]=str(destination/relative)
                atomic_write(destination/'runtime.env',''.join(k+'='+quote_env(v)+'\n' for k,v in values.items() if v is not None).encode())
                report={'state':'complete','snapshot':manifest['snapshot_id'],'lineage':manifest['lineage'],
                        'snapshot_revision':manifest['revision'],'deletion_revision':ledger['revision'],
                        'deletions_replayed':len(ledger['deletions']),'pending_jobs':pending,
                        'runtime_env':'runtime.env','asset_locations':{k:str(destination/k) for k in manifest['sources']}}
                atomic_write(destination/'restore-report.json',json_bytes(report))
                atomic_write(destination/'source-manifest.json',json_bytes(manifest))
                # The repository retains the original snapshot; do not retain another old DB copy here.
                (stage/'database.dump').unlink()
                stage.rmdir()
                marker.unlink()
                return report
            except Exception:
                atomic_write(marker,json_bytes({'state':'failed','snapshot':manifest['snapshot_id'],
                    'action':'Do not start services. Inspect this attempt; retry into a new empty database and directory.'}))
                raise
        finally: conn.close()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    commands=parser.add_subparsers(dest='command',required=True)
    for name in ('backup','restore'):
        p=commands.add_parser(name)
        p.add_argument('--dsn-env',default='SHUTTL_RECOVERY_DSN')
        p.add_argument('--pg-container',help='Optional PostgreSQL container holding pg_dump/pg_restore, port 5432')
        p.add_argument('--repository',type=Path,required=True)
        if name=='backup':
            p.add_argument('--media',type=Path,required=True);p.add_argument('--config',type=Path,required=True)
            p.add_argument('--asset',action='append',required=True,help='NAME=DIRECTORY; repeat for runtime, weights and training outputs')
        else:
            p.add_argument('--snapshot',type=Path,required=True);p.add_argument('--destination',type=Path,required=True)
            p.add_argument('--redis-env',default='SHUTTL_RECOVERY_REDIS_URL')
    p=commands.add_parser('verify');p.add_argument('snapshot',type=Path)
    args=parser.parse_args()
    try:
        if args.command=='verify':
            m=verify(args.snapshot);result={'snapshot':m['snapshot_id'],'files':len(m['files']),'verified':True}
        else:
            dsn=os.environ.get(args.dsn_env)
            if not dsn: raise RecoveryError(f'Set {args.dsn_env} to an explicit database connection string')
            database=Database(dsn,args.pg_container)
            if args.command=='backup':
                assets={}
                for value in args.asset:
                    name,sep,path=value.partition('=')
                    if not sep or name in assets: raise RecoveryError('Assets must have unique NAME=DIRECTORY labels')
                    assets[name]=Path(path)
                result=backup(database,args.repository,args.media,args.config,assets)
            else:
                redis=os.environ.get(args.redis_env)
                if not redis: raise RecoveryError(f'Set {args.redis_env} to a fresh Redis database URL')
                result=restore(database,args.repository,args.snapshot,args.destination,redis_url=redis)
        print(json.dumps(result,ensure_ascii=False,indent=2))
    except (RecoveryError,psycopg2.Error,OSError,ValueError,tarfile.TarError) as exc:
        parser.exit(1,f'Recovery failed: {exc}\n')


if __name__=='__main__': main()
