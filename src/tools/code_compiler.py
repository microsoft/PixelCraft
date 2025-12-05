import ast
import uuid
import sys
from io import StringIO

class OutputTee:
    """Duplicate stdout to both console and a buffer."""
    def __init__(self, original_stdout):
        self.stdout = original_stdout
        self.buffer = StringIO()
        
    def write(self, data):
        self.stdout.write(data)
        self.buffer.write(data)
    
    def flush(self):
        self.stdout.flush()
        self.buffer.flush()
        
    def getvalue(self):
        return self.buffer.getvalue()

def execute_code(code: str):
    try:
        tree = ast.parse(code)
    except SyntaxError as e:
        raise RuntimeError(f"SyntaxError: {e}")
    except Exception as e:
        raise RuntimeError(f"Error parsing code: {e}")

    if not tree.body:
        return None

    # check if the last statement is a print call
    last_stmt = tree.body[-1]
    is_print_call = (
        isinstance(last_stmt, ast.Expr) and
        isinstance(last_stmt.value, ast.Call) and
        isinstance(last_stmt.value.func, ast.Name) and
        last_stmt.value.func.id == 'print'
    )

    # generate a unique temporary variable name
    temp_var = f'__tmp_{uuid.uuid4().hex}__' if not is_print_call else None

    # modify the AST to assign the last expression to temp_var if it's not a print call
    if isinstance(last_stmt, ast.Expr) and not is_print_call:
        new_assign = ast.Assign(
            targets=[ast.Name(id=temp_var, ctx=ast.Store())],
            value=last_stmt.value
        )
        new_body = tree.body[:-1] + [new_assign]
        exec_tree = ast.Module(body=new_body, type_ignores=[])
    else:
        exec_tree = tree

    ast.fix_missing_locations(exec_tree)
    
    try:
        exec_code = compile(exec_tree, '<string>', 'exec')
    except Exception as e:
        raise RuntimeError(f"Error compiling code: {e}")

    env = {}
    result = None
    
    # execute the code
    try:
        if is_print_call:
            original_stdout = sys.stdout
            tee = OutputTee(original_stdout)
            sys.stdout = tee
            exec(exec_code, env)
            result = tee.getvalue().rstrip('\n')
        else:
            # normal execution mode
            exec(exec_code, env)
            result = env.get(temp_var, None)
    except Exception as e:
        raise RuntimeError(f"Error during execution: {e}")
    finally:
        if is_print_call:
            sys.stdout = original_stdout

    return result
