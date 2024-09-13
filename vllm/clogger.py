from datetime import datetime
import sys
import functools
import logging
import colorlog

clogger = logging.getLogger('color_log')
clogger.setLevel(logging.DEBUG)

clogger_handler = logging.StreamHandler()
clogger_handler.setLevel(logging.DEBUG)
clogger_handler.setFormatter(colorlog.ColoredFormatter(
    "%(log_color)s%(levelname)s %(asctime)s %(filename)s:%(lineno)d] %(message)s",
    datefmt="%m-%d %H:%M:%S",
    log_colors={
        'DEBUG': 'cyan',
        'INFO': 'green',
        'WARNING': 'yellow',
        'ERROR': 'red',
        'CRITICAL': 'bold_red',
    }))
clogger.addHandler(clogger_handler)

def init_clogger():
    return clogger


trace_logger = logging.getLogger('trace_logger')
trace_logger.setLevel(logging.INFO)
trace_logger_handler = logging.FileHandler(f"trace_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
trace_logger_handler.setLevel(logging.INFO)
trace_logger.addHandler(trace_logger_handler)

# 定义一个函数来追踪函数调用
trace_call_depth = -1
def trace_calls(frame, event, arg):
    global trace_call_depth
    code = frame.f_code
    func_file = code.co_filename  # 获取文件名
    if 'vllm' not in func_file:
        return trace_calls

    if event == 'call':  # 只追踪函数调用事件
        trace_call_depth += 1
        func_line = frame.f_lineno  # 获取行号
        func_name = code.co_name  # 获取函数名
        if 'self' in frame.f_locals: # 获得类名
            class_name = frame.f_locals['self'].__class__.__name__
        else:
            class_name = None
    
        if class_name:
            trace_logger.info('\t' * trace_call_depth + f"{class_name}.{func_name} called from file://{func_file}:{func_line}")
        else:
            trace_logger.info('\t' * trace_call_depth + f"{func_name} called from file://{func_file}:{func_line}")

    elif event == 'return':
        trace_call_depth -= 1
    return trace_calls

def trace_funcall_recursive(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        clogger.debug(f"trace function: {func.__module__} {func.__name__}")
        # start trace
        global trace_call_depth
        if trace_call_depth != -1:
            raise RuntimeError('trace_call_depth is not -1')
        sys.settrace(trace_calls)
        # call function
        result = func(*args, **kwargs)
        # end trace
        sys.settrace(None)
        return result
    return wrapper