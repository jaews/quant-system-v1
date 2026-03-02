import py_compile
import sys
try:
    py_compile.compile('app.py', doraise=True)
    print('compiled')
except py_compile.PyCompileError as e:
    print('PyCompileError:')
    print(e.msg)
    raise
except Exception as e:
    import traceback
    traceback.print_exc()
    raise
