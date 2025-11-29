'''
Global settings
'''


# Default boxes
# DEFAULT_BOXES = ((x1_offset, y1_offset, x2_offset, y2_offset), (...), ...)
# Offset is relative to upper-left-corner and lower-right-corner of the feature map cell
NUM_POINT = 2000
DEFAULT_BOXES = ((-0.5, -0.5, 0.5, 0.5))  # , (-0.2, -0.2, 0.2, 0.2))
NUM_DEFAULT_BOXES = 1  # len(DEFAULT_BOXES)
# NUM_DEFAULT_BOXES = len(DEFAULT_BOXES)

# Constants (TODO: Keep this updated as we go along)
NUM_CLASSES = 1 #femur + 1 background class
NUM_CHANNELS = 3  # grayscale->1, RGB->3
NUM_PRED_CONF = NUM_DEFAULT_BOXES * NUM_CLASSES  # number of class predictions per feature map cell
NUM_PRED_LOC = NUM_DEFAULT_BOXES * 4  # number of localization regression predictions per feature map cell

# Bounding box parameters
IOU_THRESH = 0.5  # match ground-truth box to default boxes exceeding this IOU threshold, during data prep
NMS_IOU_THRESH = 0.1  # IOU threshold for non-max suppression

# Negatives-to-positives ratio used to filter training data
NEG_POS_RATIO = 5  # negative:positive = NEG_POS_RATIO:1

# Class confidence threshold to count as detection
CONF_THRESH = 0.8

# Model selection and dependent parameters
MODEL = 'AlexNet'  # AlexNet/VGG16/ResNet50
if MODEL == 'AlexNet':
    #IMG_H, IMG_W = 360, 640
    #FM_SIZES = [[11, 19],[6, 10],[3,5]]

    # change to this for input size of [512, 512]
    IMG_H, IMG_W = 512, 512
    FM_SIZES = [[15, 15], [8, 8], [4, 4]]

else:
    raise NotImplementedError('Model not implemented')

# Model hyper-parameters
REG_SCALE = 1e-2  # L2 regularization strength
LOC_LOSS_WEIGHT = 1.  # weight of localization loss: loss = conf_loss + LOC_LOSS_WEIGHT * loc_loss

num_total_preds = 0
for fm_size in FM_SIZES:
    num_total_preds += fm_size[0] * fm_size[1] * NUM_DEFAULT_BOXES
num_total_preds_loc = num_total_preds * 4

