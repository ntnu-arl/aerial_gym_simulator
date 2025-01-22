import pyulog

import argparse
import os
import numpy as np

from pyulog.core import ULog

def convert_ulog2csv(ulog_file_name, messages, output, delimiter, time_s, time_e,
                     disable_str_exceptions=False):
    """
    Coverts and ULog file to a CSV file.

    :param ulog_file_name: The ULog filename to open and read
    :param messages: A list of message names
    :param output: Output file path
    :param delimiter: CSV delimiter
    :param time_s: Offset time for conversion in seconds
    :param time_e: Limit until time for conversion in seconds

    :return: None
    """

    msg_filter = messages.split(',') if messages else None

    ulog = ULog(ulog_file_name, msg_filter, disable_str_exceptions)
    data = ulog.data_list

    output_file_prefix = ulog_file_name
    # strip '.ulg'
    if output_file_prefix.lower().endswith('.ulg'):
        output_file_prefix = output_file_prefix[:-4]

    # write to different output path?
    if output:
        base_name = os.path.basename(output_file_prefix)
        output_file_prefix = os.path.join(output, base_name)

    for d in data:
        fmt = '{0}_{1}_{2}.csv'
        output_file_name = fmt.format(output_file_prefix, d.name.replace('/', '_'), d.multi_id)
        fmt = 'Writing {0} ({1} data points)'
        # print(fmt.format(output_file_name, len(d.data['timestamp'])))
        with open(output_file_name, 'w', encoding='utf-8') as csvfile:

            # use same field order as in the log, except for the timestamp
            data_keys = [f.field_name for f in d.field_data]
            data_keys.remove('timestamp')
            data_keys.insert(0, 'timestamp')  # we want timestamp at first position

            # we don't use np.savetxt, because we have multiple arrays with
            # potentially different data types. However the following is quite
            # slow...

            # write the header
            csvfile.write(delimiter.join(data_keys) + '\n')

            #get the index for row where timestamp exceeds or equals the required value
            time_s_i = np.where(d.data['timestamp'] >= time_s * 1e6)[0][0] \
                    if time_s else 0
            #get the index for row upto the timestamp of the required value
            time_e_i = np.where(d.data['timestamp'] >= time_e * 1e6)[0][0] \
                    if time_e else len(d.data['timestamp'])

            # write the data
            last_elem = len(data_keys)-1
            for i in range(time_s_i, time_e_i):
                for k in range(len(data_keys)):
                    csvfile.write(str(d.data[data_keys[k]][i]))
                    if k != last_elem:
                        csvfile.write(delimiter)
                csvfile.write('\n')

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description='Convert ULog to CSV')
    parser.add_argument('filename', metavar='file.ulg', help='ULog input file')

    parser.add_argument(
        '-m', '--messages', dest='messages',
        help=("Only consider given messages. Must be a comma-separated list of"
              " names, like 'sensor_combined,vehicle_gps_position'"))
    parser.add_argument('-d', '--delimiter', dest='delimiter', action='store',
                        help="Use delimiter in CSV (default is ',')", default=',')


    parser.add_argument('-o', '--output', dest='output', action='store',
                        help='Output directory (default is same as input file)',
                        metavar='DIR')
    parser.add_argument('-i', '--ignore', dest='ignore', action='store_true',
                        help='Ignore string parsing exceptions', default=False)

    parser.add_argument(
        '-ts', '--time_s', dest='time_s', type = int,
        help="Only convert data after this timestamp (in seconds)")

    parser.add_argument(
        '-te', '--time_e', dest='time_e', type=int,
        help="Only convert data upto this timestamp (in seconds)")

    args = parser.parse_args()

    if args.output and not os.path.isdir(args.output):
        print('Creating output directory {:}'.format(args.output))
        os.mkdir(args.output)
    
    dir_path = ""
    for dir in args.filename.split('/')[:-1]:
        dir_path += dir + '/'
    dir_name = args.filename.split('/')[-1].split('.')[0]
    full_path = dir_path + "/" + dir_name
    if not os.path.exists(full_path):
        os.makedirs(full_path)
        
    convert_ulog2csv(args.filename, args.messages, full_path, args.delimiter, args.time_s, args.time_e, args.ignore)
    

