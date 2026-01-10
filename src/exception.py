import sys

def get_error_details(error, error_details:sys):
    _,_,exc_tb = error_details.exc_info()
    filename = exc_tb.tb_frame.f_code.co_filename
    LineNo = exc_tb.tb_lineno
    Error = str(error)
    error_msg = "Error Occured in File {0}, Line {1}, and Error Msg is {2}".format(filename, LineNo, Error)

    return error_msg

class CustomException(Exception):

    def __init__(self, error_msg, error_details:sys):
        super().__init__(error_msg)
        self.error_msg = get_error_details(error=error_msg, error_details=error_details)
    
    def __str__(self):
        return self.error_msg