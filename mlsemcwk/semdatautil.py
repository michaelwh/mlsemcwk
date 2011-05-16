def sem_datarow_to_image(datarow):
    img = []
    for imgline in range(16):
        img.append([pix for pix in datarow[imgline*16:(imgline*16 + 16)]])
    return img

def get_sem_data(datafilepath):
    df = open(datafilepath, 'r')
    datarows = []
    labels = []
    for line in df:
        lineimg = line.rsplit(' ')
        datarows.append([float(pix) for pix in lineimg[:(15*16 + 16)]])
        labels.append(convert_datarowlabel(lineimg[(15*16 + 16):-1]))
    return datarows, labels

def convert_datarowlabel(datarowlabel):
    """the labels from the database are like this: ['1', '0', '0', '0', '0', '0', '0', '0', '0', '0'] and we want to convert them to a string"""
    for i in range(len(datarowlabel)):
        if int(datarowlabel[i]) == 1:
            return str(i)
        
def convert_to_numeric_label(label):
    i = 0
    for lab in label:
        if int(lab) == 1:
            break
        i += 1
    return i