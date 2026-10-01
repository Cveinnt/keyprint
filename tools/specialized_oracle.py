"""Bounded output checks. Python is interpreted from a tiny AST, never exec'd."""
import ast
import csv
import io
import json
import operator
import sqlite3


def python_result(text, case):
    tree = ast.parse(text)
    if len(list(ast.walk(tree))) > 160 or len(tree.body) != 1:
        raise ValueError('Require one bounded function')
    fn = tree.body[0]
    if (not isinstance(fn, ast.FunctionDef) or fn.name != case['function'] or fn.decorator_list
            or fn.args.defaults or fn.args.kwonlyargs or fn.args.vararg or fn.args.kwarg
            or fn.args.posonlyargs or [a.arg for a in fn.args.args] != case['params']):
        raise ValueError('Function contract differs')
    comparisons = {ast.Eq: operator.eq, ast.NotEq: operator.ne, ast.Lt: operator.lt,
                   ast.LtE: operator.le, ast.Gt: operator.gt, ast.GtE: operator.ge,
                   ast.Is: operator.is_, ast.IsNot: operator.is_not}

    def expression(node, env):
        if isinstance(node, ast.Constant) and type(node.value) in (bool, int):
            if abs(node.value) > 10**9: raise ValueError('Literal too large')
            return node.value
        if isinstance(node, ast.Name): return env[node.id]
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.Not):
            return not expression(node.operand, env)
        if isinstance(node, ast.IfExp):
            return expression(node.body if expression(node.test, env) else node.orelse, env)
        if isinstance(node, ast.BoolOp):
            values = [expression(n, env) for n in node.values]
            if not all(type(v) is bool for v in values): raise ValueError('Expected booleans')
            return all(values) if isinstance(node.op, ast.And) else any(values)
        if isinstance(node, ast.Compare):
            values = [expression(n, env) for n in [node.left, *node.comparators]]
            return all(comparisons[type(op)](a,b) for op,a,b in zip(node.ops,values,values[1:]))
        raise ValueError('Outside safe Python subset')

    def statements(nodes, env):
        for node in nodes:
            if isinstance(node, ast.Return): return True, expression(node.value, env)
            if isinstance(node, ast.If):
                returned, value = statements(node.body if expression(node.test, env) else node.orelse, env)
                if returned: return returned, value
            elif isinstance(node, ast.Assign) and len(node.targets)==1 and isinstance(node.targets[0], ast.Name):
                env[node.targets[0].id] = expression(node.value, env)
            else: raise ValueError('Outside safe Python subset')
        return False, None

    checks=[]
    for args, expected in case['tests']:
        returned, value = statements(fn.body, dict(zip(case['params'],args)))
        checks.append(returned and type(value) is type(expected) and value == expected)
    return all(checks), {'test_cases': len(checks), 'passed': sum(checks)}


def evaluate(text, case):
    try:
        if not isinstance(text,str) or len(text)>16000: raise ValueError('Output size')
        if case['domain']=='python':
            passed, detail=python_result(text,case)
        elif case['domain']=='sql':
            with sqlite3.connect(':memory:') as db:
                db.executescript(case['setup'])
                db.setlimit(sqlite3.SQLITE_LIMIT_LENGTH,16000)
                db.setlimit(sqlite3.SQLITE_LIMIT_SQL_LENGTH,16000)
                db.setlimit(sqlite3.SQLITE_LIMIT_EXPR_DEPTH,40)
                db.set_authorizer(lambda action,a,b,c,d: sqlite3.SQLITE_OK if action in
                    (sqlite3.SQLITE_SELECT,sqlite3.SQLITE_READ) or
                    (action==sqlite3.SQLITE_FUNCTION and b in ('sum','count','coalesce')) else sqlite3.SQLITE_DENY)
                ticks=[0]
                def progress():
                    ticks[0]+=1
                    return int(ticks[0]>100)
                db.set_progress_handler(progress,100)
                result=[list(r) for r in db.execute(text).fetchmany(101)]
                passed=result==case['expected']
                detail={'returned_rows':len(result)}
        elif case['domain']=='json':
            def unique(pairs):
                obj={}
                for key,value in pairs:
                    if key in obj: raise ValueError('Duplicate JSON key')
                    obj[key]=value
                return obj
            result=json.loads(text,object_pairs_hook=unique)
            expected=case['expected']
            passed=isinstance(result,dict) and result==expected and all(type(result[k]) is type(v) for k,v in expected.items())
            detail={'exact_typed_fields':passed}
        elif case['domain']=='csv':
            result=list(csv.reader(io.StringIO(text),strict=True))
            passed=result==case['expected']; detail={'exact_cells':passed}
        else: raise ValueError('Unknown domain')
        return {'status':'pass' if passed else 'fail','details':detail}
    except (ValueError,TypeError,KeyError,SyntaxError,sqlite3.Error,csv.Error,RecursionError) as error:
        return {'status':'unsupported_or_invalid','error_type':type(error).__name__,
                'reason':str(error)[:200]}
