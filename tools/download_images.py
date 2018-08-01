
def download_images(basedir, progress_bar=True):
    import requests
    import json
    import _init_paths
    from lib.utils.detect_notebook import is_notebook
    if is_notebook():
        from tqdm import tqdm_notebook as tqdm
    else:
        from tqdm import tqdm

    with open("{}/data/raw/paragraphs_v1.json".format(basedir), "r") as f:
        paragraph_json = json.load(f)
        for json in (tqdm(paragraph_json) if progress_bar else paragraph_json):
            resp = requests.get(json['url'], verify=False)
            with open("{}/data/images/{}.jpg".format(basedir, json['image_id']), "wb") as g:
                g.write(resp.content)

def main(args):
    download_images(args.basedir, not args.disable_progress_bar)
    print("Finished downloading!")

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--basedir', type=str, 
                        default='.', 
                        help='The root directory of the project. Default value of \'.\' (the current directory).')
    parser.add_argument('--disable_progress_bar', action='store_true', 
                        default=False, 
                        help='Set to disable the progress bar.')
    args = parser.parse_args()
    main(args)
