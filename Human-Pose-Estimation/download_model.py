import urllib.request
import os
from tqdm import tqdm
import shutil


def my_hook(t):
    last_b = [0]

    def update_to(b=1, bsize=1, tsize=None):
        if tsize is not None:
            t.total = tsize
        t.update((b - last_b[0]) * bsize)
        last_b[0] = b

    return update_to


class TqdmUpTo(tqdm):

    def update_to(self, b=1, bsize=1, tsize=None):
        if tsize is not None:
            self.total = tsize
        self.update(b * bsize - self.n)  # will also set self.n = b * bsize


def download_model():
	model_link = "http://test-1252747889.cosgz.myqcloud.com/pose_net.zip"
	eg_file = model_link.replace('/', ' ').split()[-1]
	base_dir = os.path.dirname(__file__)
	model_dir = os.path.join(base_dir, "model")
	model_path = os.path.join(model_dir, "pose_net.cntkmodel")
	if os.path.exists(model_path):
		return model_path
	if not os.path.exists(model_dir):
		os.mkdir(model_dir)
	zip_model_path = os.path.join(model_dir, model_link.split("/")[-1])
	with TqdmUpTo(unit='B', unit_scale=True, unit_divisor=1024, miniters=1,
				  desc=eg_file) as t:  # all optional kwargs
		urllib.request.urlretrieve(model_link, filename=zip_model_path, reporthook=t.update_to,
						   data=None)

	shutil.unpack_archive(zip_model_path, model_dir)
	if os.path.exists(zip_model_path):
		os.remove(zip_model_path)

	return model_path

if __name__ == "__main__":
	download_model()

