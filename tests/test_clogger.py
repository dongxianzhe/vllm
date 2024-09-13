from vllm.clogger import init_clogger

logger = init_clogger()

logger.debug('debug')
logger.info('info')
logger.warning('warning')
logger.error('error')
logger.critical('critical')