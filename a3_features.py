import os
import argparse
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from random import random
# Whatever other imports you need


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert directories into table.")
    parser.add_argument("inputdir", type=str, help="The root of the author directories.")
    parser.add_argument("outputfile", type=str, help="The name of the output file containing the table of instances.")
    parser.add_argument("dims", type=int, help="The output feature dimensions.")
    parser.add_argument("--test", "-T", dest="testsize", type=int, default="20", help="The percentage (integer) of instances to label as test.")

    args = parser.parse_args()

    print("Reading {}...".format(args.inputdir))
    # Do what you need to read the documents here.
    default_dir = r"{}\{}".format(os.getcwd(), args.inputdir)
    os.chdir(default_dir)
    data = pd.DataFrame(columns = ['label', 'value_raw'])
    for author in os.listdir():
        os.chdir(r"{}\{}".format(default_dir,author))
        for file in os.listdir():
            with open(file) as f:
                message = ""
                new_message = False
                first_message = True
                wrong_author = False
                while True:
                    current_line = f.readline()
                    if not current_line:
                        if not wrong_author:
                            data.loc[len(data), :] = [author, message]
                        break
                    if first_message:
                        if ":" in current_line:
                            continue
                        if ":" not in current_line:
                            first_message = False
                            new_message = True
                    if "-----Original Message-----" in current_line:
                        if not wrong_author:
                            data.loc[len(data), :] = [author, message]
                        new_message = True
                        wrong_author = False
                        continue
                    if new_message:
                        message = ""
                        if "From:" in current_line and author[:-2] not in current_line.lower():
                            wrong_author = True
                        elif ":" in current_line:
                            continue
                        new_message = False
                    message = message + current_line
                        
                    
    #Build the table here.
    data['value_raw'] = data['value_raw'].apply(lambda x: x.lower())
    print("Vectorizing input...")
    vect = TfidfVectorizer()
    vect.fit(data['value_raw'])
    temp = vect.transform(data['value_raw'])
    #training a PCA to reduce the number of dimensions
    if args.dims >= len(vect.vocabulary_) or args.dims <= 0:
        print("Warning: the desired number of features exceeds the actual number of distinct tokens encountered or is incorrect. Your feature vector will equal in length to the length of the list ({})".format(vect.vocabulary_))
    else:
        print("Constructing table with {} feature dimensions and {}% test instances...".format(args.dims, args.testsize))
        data['subset'] = pd.Series([random() for x in range(len(data))])
        data['subset'] = data['subset'].apply(lambda x: "test" if x < args.testsize/100 else "train")
        reductor = TruncatedSVD(args.dims)
        reductor.fit(temp)
        data.reset_index(inplace = True)
        temp = reductor.transform(temp)
        data['vector'] = data['index'].apply(lambda x: temp[x])
    print("Writing to {}...".format(args.outputfile))
    #Write the table out here.
    os.chdir(default_dir)
    data.drop('value_raw', axis = 1, inplace = True)
    data.drop('index', axis = 1, inplace = True)
    data.to_csv(args.outputfile, index = False)

    print("Done!")
    
