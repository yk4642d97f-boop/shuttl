"""Real pg_dump/pg_restore rehearsal, only in a named disposable Docker container."""
from concurrent.futures import ThreadPoolExecutor
from contextlib import closing
import fcntl
import io
import json
import os
from pathlib import Path
import subprocess
import tarfile
import time
import uuid

from dotenv import dotenv_values
import psycopg2
from psycopg2 import sql
import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import Session

from app.core.database import Base
from app.models.user import User
from app.models.match import Match, Rally, ShotPoint
from app.models.analysis_job import AnalysisJob
from app.models.media_deletion import MediaDeletion
from app.maintenance import recovery as r

DSN=os.environ.get('SHUTTL_TEST_RECOVERY_DSN')
CONTAINER=os.environ.get('SHUTTL_TEST_RECOVERY_CONTAINER')


@pytest.fixture
def env(tmp_path):
    if not DSN or not CONTAINER: pytest.skip('Set disposable SHUTTL_TEST_RECOVERY_DSN/CONTAINER')
    params=psycopg2.extensions.parse_dsn(DSN)
    assert params['host']=='127.0.0.1' and params['dbname']=='postgres'
    info=json.loads(subprocess.check_output(['docker','inspect',CONTAINER]))[0]
    assert info['Config']['Labels'].get('shuttl.test')=='recovery'
    assert any(p['HostIp']=='127.0.0.1' and p['HostPort']==params['port'] for p in info['NetworkSettings']['Ports']['5432/tcp'])
    admin=psycopg2.connect(DSN);admin.autocommit=True
    created=[]
    def new_database():
        name='shuttl_recovery_'+uuid.uuid4().hex
        with admin.cursor() as cur: cur.execute(sql.SQL('CREATE DATABASE {}').format(sql.Identifier(name)))
        created.append(name)
        return r.Database(DSN.rsplit('/',1)[0]+'/'+name,CONTAINER)
    source=new_database();target=new_database()
    engine=create_engine(source.dsn.replace("postgresql://","postgresql+psycopg2://",1))
    Base.metadata.create_all(engine)
    media=tmp_path/'media';media.mkdir()
    assets=tmp_path/'assets';(assets/'models').mkdir(parents=True)
    (assets/'models/hitnet.pt').write_bytes(b'checkpoint fixture, not trained weights')
    # A self-contained training checkpoint stands in for state/optimizer/RNG persistence.
    import torch
    generator=torch.Generator().manual_seed(7)
    checkpoint={'state':torch.randn(4,4,generator=generator),'optimizer':{'step':3},'rng':generator.get_state()}
    torch.save(checkpoint,assets/'last.pt')
    (assets/'config.json').write_text('{"steps":6,"seed":7}\n')
    (assets/'metrics.jsonl').write_text('{"step":3,"loss":1.25}\n')
    config=tmp_path/'source.env'
    config.write_text('STORAGE_BACKEND=local\nSECRET_KEY=synthetic-recovery-fixture\n'
                      f'SHUTTL_WEIGHTS_DIR={assets}/models\nSHUTTL_HITNET_WEIGHTS={assets}/models/hitnet.pt\n')
    owner=uuid.uuid4();done,pending,cancelled=[uuid.uuid4() for _ in range(3)]
    pending_job,cancelled_job=uuid.uuid4(),uuid.uuid4();token=uuid.uuid4()
    keys={m:f'videos/{owner}/{m}.mov' for m in (done,pending,cancelled)}
    for mid,key in keys.items():
        path=media/key;path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(b'fixture-video-'+str(mid).encode())
    (media/'.analysis').mkdir();(media/'.analysis/temporary.mov').write_bytes(b'exclude worker cache')
    (media/keys[done]).with_name('.'+done.hex+'.part').write_bytes(b'incomplete upload')
    with Session(engine) as db:
        db.add(User(id=owner,email='backup@test.invalid',hashed_password='fixture-password-hash'));db.flush()
        db.add_all([Match(id=done,user_id=owner,status='done',video_oss_key=keys[done],analysis={'marker':'preserve'}),
                    Match(id=pending,user_id=owner,status='analyzing',video_oss_key=keys[pending],celery_job_id=str(pending_job)),
                    Match(id=cancelled,user_id=owner,status='failed',video_oss_key=keys[cancelled],celery_job_id=str(cancelled_job))]);db.flush()
        rally=Rally(match_id=done,start_frame=10,end_frame=30,winner='near');db.add(rally);db.flush()
        db.add(ShotPoint(rally_id=rally.id,frame=20,x_px=11,y_px=22,speed_kmh=80,shot_type='smash'))
        db.add_all([AnalysisJob(id=pending_job,match_id=pending,payload={'video_oss_key':keys[pending],'calibration':{}},
                               state='running',attempt=1,claim_token=token,percent=60,stage='detect'),
                    AnalysisJob(id=cancelled_job,match_id=cancelled,payload={'video_oss_key':keys[cancelled],'calibration':{}},state='cancelled',attempt=1)])
        db.commit()
    data={'source':source,'target':target,'repo':tmp_path/'repository','media':media,'assets':assets,'config':config,
          'owner':owner,'done':done,'pending':pending,'cancelled':cancelled,'keys':keys,'pending_job':pending_job,
          'cancelled_job':cancelled_job,'token':token,'destination':tmp_path/'restored','checkpoint':checkpoint,
          'new_database':new_database}
    yield data
    engine.dispose()
    for name in created:
        assert name.startswith('shuttl_recovery_')
        with admin.cursor() as cur: cur.execute(sql.SQL('DROP DATABASE {} WITH (FORCE)').format(sql.Identifier(name)))
    admin.close()


def backup(env):
    return r.backup(env['source'],env['repo'],env['media'],env['config'],{'runtime':env['assets']})


def restore(env,snapshot):
    return r.restore(env['target'],env['repo'],snapshot,env['destination'],redis_url='redis://127.0.0.1:6379/15')


def query(database,statement,args=()):
    with closing(database.connect()) as conn,conn,conn.cursor() as cur:
        cur.execute(statement,args or None)
        return cur.fetchall() if cur.description else None


def delete_match(env,conn=None):
    owns=conn is None;conn=conn or env['source'].connect()
    with conn.cursor() as cur:
        cur.execute("INSERT INTO media_deletions(match_id,user_id,video_key,storage_backend,state,attempts,created_at,available_at) VALUES (%s,%s,%s,'local','pending',0,now(),now())",
                    (env['done'],env['owner'],env['keys'][env['done']]))
        cur.execute('DELETE FROM matches WHERE id=%s',(env['done'],))
    if owns:conn.commit();conn.close()


def test_roundtrip_restores_database_media_config_and_training_state(env):
    import torch
    result=backup(env);m=r.verify(result['snapshot'])
    assert not any('.part' in p or '.analysis/' in p for p in m['files'])
    report=restore(env,result['snapshot'])
    assert report['state']=='complete' and report['pending_jobs']==1
    assert query(env['target'],'SELECT count(*) AS n FROM matches')[0]['n']==3
    assert query(env['target'],'SELECT analysis FROM matches WHERE id=%s',(env['done'],))[0]['analysis']=={'marker':'preserve'}
    assert query(env['target'],'SELECT count(*) AS n FROM shot_points')[0]['n']==1
    job=query(env['target'],'SELECT * FROM analysis_jobs WHERE id=%s',(env['pending_job'],))[0]
    assert job['state']=='queued' and job['claim_token'] is None and job['percent']==0 and job['attempt']==1
    assert query(env['target'],'SELECT state FROM analysis_jobs WHERE id=%s',(env['cancelled_job'],))[0]['state']=='cancelled'
    assert query(env['source'],'SELECT claim_token FROM analysis_jobs WHERE id=%s',(env['pending_job'],))[0]['claim_token']==env['token']
    for key in env['keys'].values():assert (env['destination']/'media'/key).read_bytes()==(env['media']/key).read_bytes()
    recovered=torch.load(env['destination']/'assets/runtime/last.pt',weights_only=True)
    assert torch.equal(recovered['state'],env['checkpoint']['state'])
    assert torch.equal(recovered['rng'],env['checkpoint']['rng'])
    assert recovered['optimizer']=={'step':3}
    config=dotenv_values(env['destination']/'runtime.env')
    assert config['SECRET_KEY']=='synthetic-recovery-fixture'
    assert config['MEDIA_DIR']==str(env['destination']/'media')
    assert config['SHUTTL_HITNET_WEIGHTS']==str(env['destination']/'assets/runtime/models/hitnet.pt')
    assert config['DATABASE_URL']==env['target'].dsn.replace('postgresql://','postgresql+asyncpg://')
    assert (env['destination']/'runtime.env').stat().st_mode & 0o777 == 0o600
    assert not (env['destination']/'RESTORE_IN_PROGRESS.json').exists()
    cli=subprocess.run([os.sys.executable,'-m','app.maintenance.recovery','verify',result['snapshot']],capture_output=True,text=True,cwd=Path(__file__).resolve().parents[1])
    assert cli.returncode==0,cli.stderr
    assert json.loads(cli.stdout)['verified']
    assert 'synthetic-recovery-fixture' not in cli.stdout


def test_old_snapshot_replays_new_deletions_before_serving(env):
    first=backup(env)
    delete_match(env);(env['media']/env['keys'][env['done']]).unlink()
    second=backup(env)
    assert second['revision']==2
    report=restore(env,first['snapshot'])
    assert report['deletion_revision']==2 and report['snapshot_revision']==1
    assert query(env['target'],'SELECT id FROM matches WHERE id=%s',(env['done'],))==[]
    assert query(env['target'],'SELECT * FROM rallies')==[]
    assert query(env['target'],'SELECT * FROM shot_points')==[]
    assert query(env['target'],'SELECT state FROM media_deletions WHERE match_id=%s',(env['done'],))[0]['state']=='pending'
    assert not (env['destination']/'media'/env['keys'][env['done']]).exists()


def test_missing_ledger_never_restores_an_older_library(env):
    snap=backup(env);(env['repo']/'latest-deletions.json').unlink()
    with pytest.raises(r.RecoveryError,match='ledger'):restore(env,snap['snapshot'])
    assert not env['destination'].exists()
    assert query(env['target'],"SELECT * FROM pg_tables WHERE schemaname='public'")==[]


def test_database_and_destination_must_be_empty(env):
    snap=backup(env)
    query(env['target'],'CREATE TABLE treasured_data(value text)')
    query(env['target'],"INSERT INTO treasured_data VALUES ('keep')")
    with pytest.raises(r.RecoveryError,match='not empty'):restore(env,snap['snapshot'])
    assert query(env['target'],'SELECT value FROM treasured_data')==[{'value':'keep'}]
    env['destination'].mkdir();(env['destination']/'keep.txt').write_text('keep')
    with pytest.raises(r.RecoveryError,match='must not exist'):restore(env,snap['snapshot'])
    assert (env['destination']/'keep.txt').read_text()=='keep'


def test_corruption_rejected_before_database_or_directory_changes(env):
    snap=backup(env);path=Path(snap['snapshot']);raw=bytearray(path.read_bytes())
    offset=raw.index(b'fixture-video-');raw[offset]^=1;path.write_bytes(raw)
    with pytest.raises(r.RecoveryError,match='checksum'):restore(env,path)
    assert not env['destination'].exists()
    assert query(env['target'],"SELECT * FROM pg_tables WHERE schemaname='public'")==[]


def test_changed_source_is_not_published(env,monkeypatch):
    original=r.add_file
    def changing(archive,manifest,name,source):
        original(archive,manifest,name,source)
        if name=='assets/runtime/last.pt':source.write_bytes(b'changed during snapshot')
    monkeypatch.setattr(r,'add_file',changing)
    with pytest.raises(r.RecoveryError,match='changed'):backup(env)
    assert not list(env['repo'].glob('*.tar'))
    assert not (env['repo']/'latest-deletions.json').exists()


def test_active_training_and_active_worker_block_backup(env):
    lock_path=env['assets']/'.run.lock'
    with lock_path.open('ab') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        with pytest.raises(r.RecoveryError,match='Training is active'):backup(env)
    with closing(env['source'].connect()) as conn,conn.cursor() as cur:
        cur.execute('SELECT pg_advisory_lock(%s)',(r.worker_lock_key(env['pending']),))
        with pytest.raises(r.RecoveryError,match='worker is active'):backup(env)
    assert not list(env['repo'].glob('*.tar'))
    assert backup(env)['revision']==1


def test_database_freeze_blocks_final_upload_publish(env):
    with r.frozen_database(env['source']):
        with closing(env['source'].connect()) as writer,writer.cursor() as cur:
            cur.execute("SET lock_timeout='100ms'")
            with pytest.raises(psycopg2.errors.LockNotAvailable):cur.execute('SELECT * FROM matches FOR UPDATE')


def test_snapshot_observes_delete_that_committed_while_waiting_for_lock(env):
    with closing(env['source'].connect()) as setup,setup:r.lineage(setup,create=True)
    with closing(env['source'].connect()) as deleter:
        delete_match(env,deleter)
        with ThreadPoolExecutor(max_workers=1) as pool:
            future=pool.submit(backup,env)
            saw_wait=False
            for _ in range(100):
                rows=query(env['source'],"SELECT 1 FROM pg_stat_activity WHERE application_name='shuttl-recovery' AND wait_event_type='Lock' AND query LIKE 'LOCK TABLE%'")
                if rows:saw_wait=True;break
                time.sleep(.02)
            deleter.commit()  # File cleanup may lag behind this commit; restore must apply its tombstone.
            result=future.result(timeout=20)
        assert saw_wait
        m=r.verify(result['snapshot'])
        assert str(env['done']) not in [row['id'] for row in m['matches']]
        report=restore(env,result['snapshot'])
        assert report['deletions_replayed']==1


def test_recovered_outbox_executes_pending_job_without_old_broker(env):
    from app.services import analysis_jobs as jobs
    from app.services.tracking import serialize
    from app.services.tracking.pipeline_v2 import MatchAnalysis
    from app.services.tracking.data_types import VideoMeta, RallyV2, ShotEventV2, HitEvent
    import numpy as np
    snap=backup(env);restore(env,snap['snapshot'])
    delivered=[]
    jobs.dispatch_due(publisher=lambda ident:delivered.append(str(ident)),connection_factory=env['target'].connect)
    assert delivered==[str(env['pending_job'])]
    def compute(key,calibration,progress):
        assert (env['destination']/'media'/key).is_file()
        meta=VideoMeta(30.,320,180,4,np.arange(4)/30)
        analysis=MatchAnalysis(meta,rallies=[RallyV2(0,3,[ShotEventV2(HitEvent(1,1/30,(20.,30.),'near',.9),None,'smash')],winner='near')])
        return analysis,serialize.to_dict(analysis)
    result=jobs.execute(env['pending_job'],compute=compute,connection_factory=env['target'].connect)
    assert result['status']=='done'
    job=query(env['target'],'SELECT * FROM analysis_jobs WHERE id=%s',(env['pending_job'],))[0]
    assert job['claim_token']!=env['token'] and job['attempt']==2
    assert query(env['target'],'SELECT status FROM matches WHERE id=%s',(env['pending'],))[0]['status']=='done'


def test_wrong_repository_lineage_cannot_merge_unrelated_accounts(env):
    snap=backup(env);ledger=json.loads((env['repo']/'latest-deletions.json').read_text());ledger['lineage']=str(uuid.uuid4())
    (env['repo']/'latest-deletions.json').write_text(json.dumps(ledger))
    with pytest.raises(r.RecoveryError,match='different database'):backup(env)
    with pytest.raises(r.RecoveryError,match='another database'):restore(env,snap['snapshot'])


def test_models_outside_asset_roots_and_missing_videos_fail(env):
    env['config'].write_text('STORAGE_BACKEND=local\nSHUTTL_HITNET_WEIGHTS='+str(env['config'])+'\n')
    with pytest.raises(r.RecoveryError,match='outside all'):backup(env)
    env['config'].write_text('STORAGE_BACKEND=local\n')
    (env['media']/env['keys'][env['done']]).unlink()
    with pytest.raises(r.RecoveryError,match='missing its source video'):backup(env)


@pytest.mark.parametrize('kind',['traversal','absolute','symlink','duplicate'])
def test_unsafe_archives_are_rejected(tmp_path,kind):
    archive=tmp_path/'bad.tar'
    with tarfile.open(archive,'w') as tar:
        item=tarfile.TarInfo('../outside' if kind=='traversal' else '/outside' if kind=='absolute' else 'payload')
        item.mode=0o600;item.size=0
        if kind=='symlink':item.type=tarfile.SYMTYPE;item.linkname='/tmp/outside'
        tar.addfile(item,io.BytesIO())
        if kind=='duplicate':tar.addfile(item,io.BytesIO())
    with pytest.raises(r.RecoveryError):r.verify(archive)


def test_symlink_and_overlapping_sources_are_rejected(tmp_path):
    root=tmp_path/'state';root.mkdir();media=root/'media';media.mkdir()
    config=tmp_path/'config.env';config.write_text('STORAGE_BACKEND=local\n')
    database=r.Database('postgresql://unused:unused@127.0.0.1/unused')
    with pytest.raises(r.RecoveryError,match='overlap'):r.backup(database,tmp_path/'repo',media,config,{'all':root})
    link=tmp_path/'link';link.symlink_to(root,target_is_directory=True)
    with pytest.raises(r.RecoveryError,match='symlink'):r.backup(database,tmp_path/'repo',media,config,{'all':link})


def test_real_bst_resume_after_relocation_matches_uninterrupted_training(env):
    """Actual released model + existing mechanics-smoke clips, not a model quality test."""
    import hashlib
    import shutil
    import torch
    project=Path(__file__).resolve().parents[2]
    research=project/'research/cloud_upgrade'
    required=[research/'cloud/run_bst.py',research/'cloud_runs/bst_smoke/data',
              research/'artifacts/bst_seq30/bst_CG_AP_JnB_bone_merged.pt',research/'cloud_runs/bst_smoke/full/last.pt']
    if not all(p.exists() for p in required):pytest.skip('Real BST fixture artifacts are not installed')
    payload=env['assets']/'project'
    shutil.copytree(project/'backend/app',payload/'backend/app',ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
    cloud=payload/'research/cloud_upgrade/cloud';cloud.mkdir(parents=True)
    shutil.copy2(required[0],cloud/'run_bst.py')
    shutil.copytree(required[1],payload/'data')
    shutil.copy2(required[2],payload/'initial.pt')
    def train(root,extra):
        command=[os.sys.executable,str(root/'research/cloud_upgrade/cloud/run_bst.py'),'train',
            '--data',str(root/'data/manifest.json'),'--init',str(root/'initial.pt'),'--out',str(root/'training'),
            '--steps','6','--batch','4','--eval-every','3','--hours','.2',*extra]
        process=subprocess.run(command,cwd=root,env={**os.environ,'PYTHONPATH':''},capture_output=True,text=True,timeout=120)
        assert process.returncode==0,process.stdout+process.stderr
        return process.stdout
    interrupted_log=train(payload,['--stop-after','3'])
    original_hash=hashlib.sha256((payload/'training/last.pt').read_bytes()).hexdigest()
    snap=backup(env)
    # Source checkpoint/data/code disappear from their former location before restore/resume.
    env['assets'].rename(env['assets'].with_name('retired-source'))
    restore(env,snap['snapshot'])
    newroot=env['destination']/'assets/runtime/project'
    restored_hash=hashlib.sha256((newroot/'training/last.pt').read_bytes()).hexdigest()
    assert original_hash==restored_hash and not payload.exists()
    resumed_log=train(newroot,['--resume'])
    full=torch.load(required[3],map_location='cpu',weights_only=True)
    recovered=torch.load(newroot/'training/last.pt',map_location='cpu',weights_only=True)
    delta=max(float((full['state'][k]-recovered['state'][k]).abs().max()) for k in full['state'])
    assert delta==0 and full['step']==recovered['step']==6
    assert full['history']==recovered['history']
    assert torch.equal(full['rng']['cpu'],recovered['rng']['cpu'])
    report={'scope':'CPU backup, relocation and BST resume only; no model quality claim',
            'interruption_step':3,'final_step':6,'max_parameter_error':delta,'validation_history_equal':True,
            'rng_equal':True,'checkpoint_sha256_before_and_after_restore':original_hash,
            'reference_checkpoint_sha256':hashlib.sha256(required[3].read_bytes()).hexdigest(),
            'trainer_sha256':hashlib.sha256(required[0].read_bytes()).hexdigest(),
            'snapshot_verified_files':len(r.verify(snap['snapshot'])['files']),
            'original_training_path_absent':not payload.exists(),
            'logs':{'interrupted':interrupted_log,'resumed':resumed_log}}
    if os.environ.get('SHUTTL_RECOVERY_TRAINING_REPORT'):
        Path(os.environ['SHUTTL_RECOVERY_TRAINING_REPORT']).write_text(json.dumps(report,indent=2)+'\n')


def test_post_dump_failure_leaves_explicit_failed_marker_and_no_runtime_config(env,monkeypatch):
    snap=backup(env)
    def fail(*args): raise r.RecoveryError('injected replay failure')
    monkeypatch.setattr(r,'replay_deletions',fail)
    with pytest.raises(r.RecoveryError,match='injected'):restore(env,snap['snapshot'])
    marker=json.loads((env['destination']/'RESTORE_IN_PROGRESS.json').read_text())
    assert marker['state']=='failed'
    assert not (env['destination']/'runtime.env').exists()
    assert not (env['destination']/'restore-report.json').exists()
    assert query(env['source'],'SELECT count(*) AS n FROM matches')[0]['n']==3


def test_stale_and_corrupt_deletion_ledgers_are_not_accepted(env):
    backup(env);newer=backup(env)
    path=env['repo']/'latest-deletions.json';ledger=json.loads(path.read_text())
    ledger['revision']=1;path.write_text(json.dumps(ledger))
    with pytest.raises(r.RecoveryError,match='stale'):restore(env,newer['snapshot'])
    path.write_text('{"interrupted":')
    with pytest.raises(r.RecoveryError,match='damaged'):backup(env)
    assert path.read_text()=='{"interrupted":'
    assert not env['destination'].exists()
