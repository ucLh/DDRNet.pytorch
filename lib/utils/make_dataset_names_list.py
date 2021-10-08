from os import walk, path

images = []
labels = []

for (dirpath, dirnames, filenames) in walk('/media/hdd/benchmarks/Mapillary_v1.2/validation/images'):
    filenames = sorted(filenames)
    images = list(map(lambda x: path.join(dirpath, x), filenames))

for (dirpath, dirnames, filenames) in walk('/media/hdd/benchmarks/Mapillary_v1.2/validation/instances'):
    filenames = sorted(filenames)
    labels = list(map(lambda x: path.join(dirpath, x), filenames))

with open('../../data/list/mappilary/val.lst', 'w') as f:
    assert len(labels) == len(images)
    for i in range(len(labels)):
        f.write(images[i] + '\t' + labels[i] + '\n')
    
