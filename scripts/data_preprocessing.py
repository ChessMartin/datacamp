import datetime as dt
import calendar
import pandas as pd
import numpy as np
import argparse
import re

def generate_dataset(f):
        df = pd.read_csv(f)
        df.columns = ['id', 'timestamp', 'long', 'lat']
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        print(df)
        print(df.dtypes)

def open_file(input_file_path):
    with open(input_file_path) as f:
        generate_dataset(f)

def main():
    # Initialize parser
    parser = argparse.ArgumentParser()

    # Adding optional argument
    parser.add_argument("-i", "--Input", help="Input File Path")
    parser.add_argument("-o", "--Output", help="Output File Path")

    # Read arguments from command line
    args = parser.parse_args()
    if args.Input:
        print("Displaying Output as: % s" % args.Input)

    # Iterating through the files to create new dataset
    open_file(args.Input)


if __name__ == '__main__':
    main()
