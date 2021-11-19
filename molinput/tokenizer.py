def createTokenizer(item):
    if type(item)==bool:
        temp = TypeTokenizer(int)
        return temp
    elif type(item)==int:
        return IdentityTokenizer()
    elif type(item)==float:
        return IdentityTokenizer()
    elif type(item)==list:
        return ListTokenizer()
    return Tokenizer()

class Tokenizer(object):
    """ A class to turn arbitrary inputs into integer classes. """
    def __init__(self):
        # the default class for an unseen entry during test-time
        self._data = {'unk': 0}
        self.num_classes = 1
        self.invert = {0: 'unk'}
        self.train = True
        self.unknown = []

    def __call__(self, item):
        """ Check to see if the Tokenizer has seen `item` before, and if so,
        return the integer class associated with it. Otherwise, if we're
        training, create a new integer class, otherwise return the 'unknown'
        class.

        """
        try:
            return self._data[item]

        except KeyError:
            if self.train:
                self._add_token(item)
                return self(item)

            else:
                # Record the unknown item, then return the unknown label
                self.unknown += [item]
                return self._data['unk']

    def get_invert(self,item):
        return self.invert[item]

    def _add_token(self, item):
        self.num_classes += 1
        self._data[item] = self.num_classes-1
        self.invert[self.num_classes-1] = item

    def __len__(self):
        return len(self._data)

class IdentityTokenizer(object):
    def __init__(self):
        pass
    def __call__(self, item):
        return item

class TypeTokenizer(object):
    def __init__(self,ref_type):
        if not type(ref_type)==type:
            raise TypeError("ref_type should be of class type.")
        self.type = ref_type
    def __call__(self,item):
        return self.type(item)

class ListTokenizer(object):
    def __init__(self):
        self.internal_tokenizer = Tokenizer()
    def __call__(self,item):
        temp = ".".join([str(x) for x in item])
        return self.internal_tokenizer(temp)
