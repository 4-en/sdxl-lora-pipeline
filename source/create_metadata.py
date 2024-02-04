# create a metadata.csv file for the dataset

import os

def to_csv_field(text: str, seperator: str=',', always_escape=False) -> str:
    """
    Convert a string to a csv field
    :param text: string
    :param seperator: seperator to be used in the csv field
    :return: csv field
    """
    text = text.replace('"', '""')

    if (always_escape or seperator in text) and not (text.startswith('"') and not text.startswith('""')):
        return f'"{text}"'
    return text


def to_csv_row(columns: list, seperator: str=',') -> str:
    """
    Convert a list of strings to a csv row
    :param columns: list of strings
    :param seperator: seperator to be used in the csv row
    :return: csv row
    """
    return seperator.join([to_csv_field(column, seperator, (i!=0)) for i, column in enumerate(columns)])
    

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
            row = to_csv_row([filename, text])
            f.write(row + '\n')

    print(f'metadata.csv file created in {directory}')

if __name__ == '__main__':
    create_metadata('data/fruits', 'foo bar')
    