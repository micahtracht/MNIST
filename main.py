'''
DISCLAIMER
This section was written largely with the use of ChatGPT. I did not write this myself.
'''

import sys
import runpy

def main():
    print("Which MNIST algorithm would you like to run?")
    print("  1) kmeans")
    print("  2) least_squares")
    print("  3) cnn")
    choice = input("Enter 1, 2, 3, or name: ").strip().lower()

    if choice in ('1', 'kmeans'):
        runpy.run_path('kmeans.py', run_name='__main__')
    elif choice in ('2', 'least_squares', 'leastsquares'):
        runpy.run_path('least_squares.py', run_name='__main__')
    elif choice in ('3', 'cnn'):
        runpy.run_path('convNN.py', run_name='__main__')
    else:
        print(f"❌  Invalid choice: {choice}")
        sys.exit(1)

if __name__ == '__main__':
    main()
