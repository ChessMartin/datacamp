import os
import sys
import argparse
import pandas as pd
import datetime as dt
from geopy.distance import distance
import math

def generate_dataset(f):
        df = pd.read_csv(f)
        df.columns = ['id', 'timestamp', 'lon', 'lat']
        df['timestamp'] = pd.to_datetime(df['timestamp'])

        vectors = pd.DataFrame(columns=['distance', 'deltatime', 'theta',
                                        'lat', 'lon', 'speed'])
        for idx in range(len(df) - 1):
            #Distance between points
            d = distance(
                    (df.iloc[idx]['lat'],     df.iloc[idx]['lon']),
                    (df.iloc[idx + 1]['lat'], df.iloc[idx + 1]['lon'])
                ).km

            #Time between points
            t = (df.iloc[idx + 1]['timestamp'] - df.iloc[idx]['timestamp']
                ).total_seconds() / 3600

            #speed in km/h between points
            s = d / t if t != 0 else 0
            if s < 3 or s > 200:
                continue

            #Angle between points
            theta = math.atan2(
                    df.iloc[idx + 1]['lat'] - df.iloc[idx]['lat'],
                    df.iloc[idx + 1]['lon'] - df.iloc[idx]['lon']
                )

            #Initial points
            lat = df.iloc[idx]['lat']
            lon = df.iloc[idx]['lon']
            vectors.loc[-1] = [d, t, theta, lat, lon, s]
            vectors.index += 1

        return vectors

def open_file(input_file_path, output_file_path):
    with open(input_file_path, 'r') as f:
        df = generate_dataset(f)
        with open(output_file_path, 'a') as out_f:
            df.to_csv(out_f, index=False)

def open_folder(input_folder_path, output_file_path, max_files):
    print("Loading {} files".format(max_files))
    for f_idx, filename in enumerate(os.listdir(input_folder_path)):
        sys.stdout.write("\rReading file: {}".format(f_idx))
        sys.stdout.flush()
        if f_idx > max_files:
            break

        with open(os.path.join(input_folder_path, filename), 'r') as f:
            df = generate_dataset(f)
            with open(output_file_path, 'a') as out_f:
                df.to_csv(out_f, index=False)

def main():
    # Initialize parser
    parser = argparse.ArgumentParser()

    # Adding optional argument
    parser.add_argument("-i", "--Input_file", help="Input File Path")
    parser.add_argument("-f", "--Input_folder", help="Input folder Path")
    parser.add_argument("-o", "--Output", help="Output File Path")
    parser.add_argument("-m", "--Max_files", help="Number of files maximum to read")

    # Read arguments from command line
    args = parser.parse_args()
    output_file = args.Output if args.Output else "./dataset.csv"
    max_files = int(args.Max_files) if args.Max_files else 100
    if args.Input_file:
        print("Displaying Output as: % s" % args.Input_file)
        open_folder(args.Input_file, output_file)

    elif args.Input_folder:
        print("Displaying Output as: % s" % args.Input_folder)
        open_folder(args.Input_folder, output_file, max_files)

    # Iterating through the files to create new dataset


if __name__ == '__main__':
    main()

