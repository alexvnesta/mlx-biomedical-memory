import subprocess
import sys

from engram_poc.resource_guard import acquire_model_lease


def test_model_lease_rejects_overlap_and_releases_on_close(tmp_path):
    path=tmp_path/'model.lock';lease=acquire_model_lease(path)
    code='import sys; from engram_poc.resource_guard import acquire_model_lease; h=acquire_model_lease(sys.argv[1])'
    result=subprocess.run([sys.executable,'-c',code,str(path)],capture_output=True,text=True)
    assert result.returncode!=0 and 'holds the model lease' in result.stderr
    lease.close()
    result=subprocess.run([sys.executable,'-c',code,str(path)],capture_output=True,text=True)
    assert result.returncode==0,result.stderr
