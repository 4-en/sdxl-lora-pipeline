# create a metadata.csv file for the dataset

import os

def create_metadata(directory:str, text) -> None:
    """
    Create a metadata.csv file for the dataset
    :param directory: path to the directory where the image files are stored
    :param text: text to be written to the metadata.csv in the text column
    :return: None
    """

    # get all filenames in the directory
    filenames = os.listdir(directory)

    formats = ['jpg', 'jpeg', 'png', 'gif', 'bmp', 'tiff', 'webp']

    # filter out non-image files
    filenames = [filename for filename in filenames if filename.split('.')[-1].lower() in formats]

    with open(directory + '/metadata.csv', 'w') as f:
        f.write('file_name,text\n')
        for filename in filenames:
            f.write(f'{filename},{text}\n')

    print(f'metadata.csv file created in {directory}')

if __name__ == '__main__':
    create_metadata('data/fruits', 'fruits123')
    