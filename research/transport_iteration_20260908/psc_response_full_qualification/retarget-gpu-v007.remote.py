import subprocess,json
job='45612641';before=subprocess.check_output(['scontrol','show','job',job],text=True);d={'before':before,'requested_node':'v007','same_job':job,'new_submission':False}
if 'JobState=PENDING' in before:
 r=subprocess.run(['scontrol','update','JobId='+job,'ReqNodeList=v007'],capture_output=True,text=True);d.update(update_returncode=r.returncode,update_stdout=r.stdout,update_stderr=r.stderr)
else:d['mutation_skipped']='no longer pending'
d['after']=subprocess.check_output(['scontrol','show','job',job],text=True);print(json.dumps(d))
