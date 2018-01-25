class BadModelException(Exception):

    def __init__(self, message):
        super(BadModelException, self).__init__(message)
