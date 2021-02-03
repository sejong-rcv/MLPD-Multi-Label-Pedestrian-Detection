import subprocess, atexit

def run_tensorboard( jobs_dir, port=6006 ):
    pid = subprocess.Popen( ['tensorboard', '--logdir', jobs_dir, '--host', '0.0.0.0', '--port', str(port)] )    
    
    def cleanup():
    	pid.kill()

    atexit.register( cleanup )
