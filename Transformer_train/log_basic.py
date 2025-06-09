import logging

def __init__(level=logging.DEBUG, file=None):
    if file != None:
        logging.basicConfig(
            level=level,
            format="%(asctime)s [%(levelname)s] %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
            filename=file,
            filemode='w',
            )
    else:
        logging.basicConfig(
            level=level,
            format="%(asctime)s \33[36m[%(levelname)s] \33[0m%(message)s",
            datefmt="\33[35m%Y-%m-%d \33[32m%H:%M:%S",
            )
    logging.info("log initialization complete")
    
if __name__ == "__main__": __init__()