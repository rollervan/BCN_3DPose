#!/usr/bin/python

import sys
from train import BCN

def main():
    # print command line arguments
    case =  sys.argv[1:][0]
    if case == 'train':
        restore = False
        s = BCN(restore=restore)
        s.train()
    elif case == 'test':
        restore = True
        s = BCN(restore)
        s.test()
    elif case ==  'test_video':
        restore = True
        s = BCN(restore)
        s.predict_video()

    print case

if __name__ == "__main__":
    main()