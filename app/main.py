from preprocessing.funnel import Funnel
from preprocessing.prepro import PreProcessor

def test():
    p = PreProcessor()
    f = Funnel(p)

    f.hello_world()

if __name__ == "__main__":
    test()