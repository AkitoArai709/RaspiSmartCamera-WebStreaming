"""buffer.py
    Summay: 
        Buffer management.
"""

class Buffer:
    """ Buffer management class.
    """

    def __init__(self, size):
        """ Initialise buffer.

        Args:
            size (ing): Max buffer size.
        """
        self.maxSize = size
        self.buffer = []
    
    def push(self, item: int):
        """ Push in buffer.

        Args:
            item (int): The value to store in the buffer 
        """
        if self.size() >= self.maxSize:
            self.buffer.pop(0)
        self.buffer.append(item)
    
    def pop(self):
        """ Pop in buffer.

        Returns:
            int: The value stored in the buffer 
        """
        ret = None
        if self.size() != 0:
            ret = self.buffer.pop(0)
        return ret

    def size(self):
        """ Current number of stored in buffer.

        Returns:
            int: Current buffer size.
        """
        return len(self.buffer)
    
    def getAvg(self):
        """ Get the average value in the buffer.

        Returns:
            int: The average value in the buffer
        """
        return sum(self.buffer)/self.size()