import traceback, sys

ex_type, ex_value, ex_traceback = sys.exc_info()            

# Extract unformatter stack traces as tuples
trace_back = traceback.extract_tb(ex_traceback)

# Format stacktrace
stack_trace = list()

for trace in trace_back:
    stack_trace.append("File : %s , Line : %d, Func.Name : %s, Message : %s" % (trace[0], trace[1], trace[2], trace[3]))

sys.stderr.write("[Error] Exception type : %s \n" % ex_type.__name__)
sys.stderr.write("[Error] Exception message : %s \n" %ex_value)
for trace in stack_trace:
        sys.stderr.write("[Error] (Stack trace) %s\n" % trace)