# -*- coding: utf-8 -*-
import logging, os
logging.getLogger("matplotlib").setLevel(logging.WARNING)
def getLogger( logFile,level=logging.DEBUG):
    os.makedirs(os.path.dirname(logFile), exist_ok=True)
    logger = logging.getLogger()
    logger.setLevel(level)
    if logger.hasHandlers():
        logger.handlers.clear()
    
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')    
    formatter_brief = logging.Formatter('%(message)s')   
    
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.DEBUG)  
    console_handler.setFormatter(formatter_brief)
    
    file_handler = logging.FileHandler(logFile, mode='w')
    file_handler.setLevel(logging.DEBUG)    
    file_handler.setFormatter(formatter)
    
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)
    return logger






