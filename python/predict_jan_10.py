# coding: utf-8
#Sylvain Kritter 10 January 2017
import os
from PIL import Image
import dicom
import scipy as sp
from scipy import ndimage
import skimage
from skimage import measure
import numpy as np
import cnn_model as CNN4

os.environ['KERAS_BACKEND'] = 'theano'
#from keras.models import model_from_json
from keras import backend as K
K.set_image_dim_ordering('th')
from keras.models import load_model

print('SUCCESS:   this is the predict.py file answering')

#########################################################
# add path of where flask should look
flaskapp_dir = '/var/www/html/flaskapp'
#flaskapp_dir = '/Users/peterhirt/googleCloud/server-ml/flaskapp'
#flaskapp_dir = 'C:/Users/sylvain/Documents/boulot/startup/radiology/web/microserver-ml/flaskapp'
#
picklefile_source = 'pickle_source'  # subdirectory name to collect weights
#picklefile_source = 'pickle_ex54'
# patch size in pixels example 32 * 32
dimpavx = 16
dimpavy = 16

namedirtop = 'predict_file'   # global directory for predict file

lungmask = 'lung_mask'        # directory with lung mask dicom
lungmask_bmp = 'bmp'  # extension for lung mask_bmp if exists
picklein_file = os.path.join(flaskapp_dir, picklefile_source)
# alternate way, probably better
here = os.path.dirname(__file__)
print('the flask app is under: ', flaskapp_dir, here)

# parameters for tools

thrpatch = 0.9              # threshold for patch acceptance in area ratio
thrprobaUIP = 0.5  # probability to take into account patches for UI
thrproba = 0.5  # probability to take into account patches for visu
avgPixelSpacing = 0.734   # average pixel spacing
subErosion = 15  # erosion factor for subpleura in mm
# imageDepth=255 #number of bits used on dicom images (2 **n) 8 bits
imageDepth = 8191  # number of bits used on dicom images (2 **n) 13 bits

#
writeFile = False  # allows to write file with UIP volume calculation
volumeweb = 'volumeweb'  #directory  where to put volume data text file if writeFile is True

pxy = float(dimpavx * dimpavy)
##########################################################################
# END OF PARAMETERS
##########################################################################
print('the current working directory is : ', flaskapp_dir)
errorDetected = False

classif = {
    'back_ground': 0,
    'consolidation': 1,
    'HC': 2,
    'ground_glass': 3,
    'healthy': 4,
    'micronodules': 5,
    'reticulation': 6,
    'air_trapping': 7,
    'cysts': 8,
    'bronchiectasis': 9,

    'bronchial_wall_thickening': 10,
    'early_fibrosis': 11,
    'emphysema': 12,
    'increased_attenuation': 13,
    'macronodules': 14,
    'pcp': 15,
    'peripheral_micronodules': 16,
    'tuberculosis': 17
}
usedclassifUIP = {
    'back_ground': 0,
    'consolidation': 1,
    'HC': 2,
    'ground_glass': 3,
    'healthy': 4,
    'micronodules': 5,
    'reticulation': 6,
    'air_trapping': 7,
    'cysts': 8,
    'bronchiectasis': 9,
}
#########################################################


def normi(tabi):
    """ normalise patches 0 imageDepth"""

    max_val = float(np.max(tabi))
    min_val = float(np.min(tabi))

#     print 'tabi1',min_val, max_val,imageDepth/float(max_val)
    tabi2 = (tabi - min_val) * (imageDepth / (max_val - min_val))
    if imageDepth < 256:
        tabi2 = tabi2.astype('uint8')
    else:
        tabi2 = tabi2.astype('uint16')
    return tabi2


def rsliceNum(s, c, e):
    ''' look for  afile according to slice number'''
    # s: file name, c: delimiter for snumber, e: end of file extension
    endnumslice = s.find(e)
    posend = endnumslice
    while s.find(c, posend) == -1:
        posend -= 1
    debnumslice = posend + 1
    return int((s[debnumslice:endnumslice]))


def genebmp(datascan, datalung,path_patient):
    global errorDetected
    """
    1.) loads the specified DICOM file from the specified folder
    2.) saves the DICOM file as np array
    3.) checks if there are DICOM files in the lung_mask folder
    4.) Depending on outcome of 3, saves a np array version of the DICOM file
    or creates a faked version with all values being at imageDepth
    """
    if not os.path.exists(datascan):
        print ('ERROR no slice number :', datascan, ' in :', path_patient)
        errorDetected = True
        return

    print ('load scan dicom file in :', datascan)

    RefDs = dicom.read_file(datascan)
    dsr = RefDs.pixel_array
    dsr = dsr - dsr.min()

    if imageDepth < 256:
        dsr = dsr.astype('uint8')
    else:
        dsr = dsr.astype('uint16')
    # resize the dicom to have always the same pixel/mm
    fxs = float(RefDs.PixelSpacing[0]) / avgPixelSpacing

    dsrresize = sp.misc.imresize(dsr, fxs, interp='bilinear', mode=None)
    # scale the dicom pixel range to be in 0-imageDepth
    dsrresize = normi(dsrresize)
    # calculate the  new dimension of scan image
    dimtabx = int(dsrresize.shape[0])
    dimtaby = int(dsrresize.shape[1])
    print ('load lung file in :', datalung)
    if datalung != 'XX':
        if datalung.find('.dcm') > 0:
            RefDslung = dicom.read_file(datalung)
            fxslung = float(RefDslung.PixelSpacing[0]) / avgPixelSpacing
            dsrLung = RefDslung.pixel_array
            dsrLung = dsrLung - dsrLung.min()
            c = float(255) / dsrLung.max()
            dsrLung = dsrLung * c
            dsrLung = dsrLung.astype('uint8')
#            cv2.imshow('erosion',erosion)

            listlungdict = sp.misc.imresize(
                dsrLung, fxslung, interp='bilinear', mode=None)

        else:
            im = Image.open(datalung).convert('L')
            listlungdict = np.array(im)
    else:
        listlungdict = np.ones((dimtabx, dimtaby), dtype='uint8')

    return dsrresize, listlungdict, dimtabx


def generate_patches(data, img1, imglung):
    """ generate patches from slice"""
    print('generate patches on: ', data)
    patch_list = []
    patch_area = []
    imglungcopy = np.copy(imglung)
    np.putmask(imglungcopy, imglungcopy == 1, 0)
    np.putmask(imglungcopy, imglungcopy > 0, 1)

    atabf = np.nonzero(imglungcopy)
    imagemax = np.count_nonzero(imglungcopy)

    if imagemax > 0:
        xmin = atabf[1].min()
        xmax = atabf[1].max()
        ymin = atabf[0].min()
        ymax = atabf[0].max()

    else:
        xmin = 0
        xmax = 20
        ymin = 0
        ymax = 20

    x = xmin
    nbp = 0
    while x <= xmax:
        y = ymin
        while y <= ymax:
            crop_img = imglungcopy[y:y + dimpavy, x:x + dimpavx]
            # convention img[y: y + h, x: x + w]

            area = crop_img.sum()

            targ = float(area) / pxy

            if targ > thrpatch:

                crop_img_orig = img1[y:y + dimpavy, x:x + dimpavx]

                imagemax = np.count_nonzero(crop_img_orig)
                im = Image.fromarray(crop_img_orig, 'L')
#                print 'imagemax',imagemax
                if imagemax > 0:
                    min_val = im.getextrema()[0]
                    max_val = im.getextrema()[1]
                else:
                    min_val = 0
                    max_val = 0
                if imagemax > 0 and max_val - min_val > 10:

                    nbp += 1
                    # append the patch coordinate to patch_list
                    patch_list.append((x, y))
                    # append the patch table image to patch_area
                    patch_area.append(crop_img_orig)
                # we put all 0 the source
                    imglungcopy[y:y + dimpavy, x:x + dimpavx] = 0
                    y += dimpavy - 1

            y += 1
        x += 1

    return patch_list, patch_area


def ILDCNNpredict(pa):
    print ('Predict started ....')
    # adding a singleton dimension and rescale to [0,1]
    pa = np.asarray(np.expand_dims(pa, 1)) / float(imageDepth)

    # look if the predict source is empty
    X0 = pa.shape[0]
    # predict and store  classification and probabilities if not empty
    if X0 > 0:
        proba = model.predict_proba(pa, batch_size=100)

    else:
        print (' no patch in selected slice')
        proba = ()
    print 'number of patches', len(pa)

    return proba


def modelCompilation():
    global errorDetected
    modelpath = os.path.join(picklein_file, 'ILD_CNN_model.h5')

    if os.path.exists(modelpath):
        model = load_model(modelpath)
        model.compile(optimizer='Adam', loss=CNN4.get_Obj('ce'))
        return model
    else:
        print ('ERROR model and weights do not exist:', modelpath)
        errorDetected = True


def definePatientSet(pp, lung_dir, lung_dir_bmp):
    global errorDetected
    # list of path to scan files and lung  and scanNumber
    # (scan,lung,ScanNumber)
    ldcm = []
    upperset = []
    middleset = []
    lowerset = []
    allset = []
    lungSegment = {}
    # list how many slice number and which one

    listscanfile = [name for name in os.listdir(pp) if
                    name.lower().find('.dcm', 0) > 0]
    if len(listscanfile) == 0:
        errorDetected = True
        return ldcm, lungSegment
    lscan = []  # store couple scan file, scan number
    for ll in listscanfile:
        path_scan = os.path.join(pp, ll)
        RefDs = dicom.read_file(path_scan)
        scanNumber = int(RefDs.InstanceNumber)
        lscan.append((scanNumber, path_scan))

    if os.path.exists(lung_dir_bmp):
        llung = []  # store couple lung file, lung number for bmp
        print 'using bmp for lung'
        listscanlung = [name for name in os.listdir(lung_dir_bmp) if
                        name.lower().find('.bmp', 0) > 0]
        for ll in listscanlung:
            scanNumber = rsliceNum(ll, '_', '.bmp')
            path_lung = os.path.join(lung_dir_bmp, ll)
            llung.append((scanNumber, path_lung))

    elif os.path.exists(lung_dir):
        llung = []  # store couple lung file, lung number for dcm
        print 'using dcm for lung'
        listscanlung = [name for name in os.listdir(lung_dir) if
                        name.lower().find('.dcm', 0) > 0]

        for ll in listscanlung:
            path_lung = os.path.join(lung_dir, ll)
            RefDs = dicom.read_file(path_lung)
            scanNumber = int(RefDs.InstanceNumber)
            llung.append((scanNumber, path_lung))
    else:
        listscanlung = []

    path_lung = 'XX'
    for ls in lscan:
        for ll in llung:
            if ls[0] == ll[0]:
                ldcm.append((ls[1], ll[1], ll[0]))
                break

    numberFile = len(ldcm)
    Nset = numberFile / 3
    print 'total number of scans: ', numberFile, 'in each set: ', Nset
    for scanumber in range(0, numberFile):
        allset.append(ldcm[scanumber][2])
        if scanumber < Nset:
            upperset.append(ldcm[scanumber][2])
        elif scanumber < 2 * Nset:
            middleset.append(ldcm[scanumber][2])
        else:
            lowerset.append(ldcm[scanumber][2])
        lungSegment['upperset'] = upperset
        lungSegment['middleset'] = middleset
        lungSegment['lowerset'] = lowerset
        lungSegment['allset'] = allset
    print 'lung segment ', lungSegment
    return ldcm, lungSegment


def posP(sln, lungSegment):
    '''define where is the slice number'''
    if sln in lungSegment['upperset']:
        psp = 'upperset'
    elif sln in lungSegment['middleset']:
        psp = 'middleset'
    else:
        psp = 'lowerset'
    return psp


def myarea(vs):
    a = 0
    x0, y0 = vs[0]
    for [x1, y1] in vs[1:]:
        dx = x1 - x0
        dy = y1 - y0
        a += 0.5 * (y0 * dx - x0 * dy)
        x0 = x1
        y0 = y1
    return a


def calcMed(scanfile, lungfile, slicenum, tabMed, dimtabx):
    '''calculate the median position in between left and right lung'''
    print 'calculate the position between lung for: ', slicenum
    ke = 5
    erosion = ndimage.grey_erosion(lungfile, size=(ke, ke))
    dilation = ndimage.grey_dilation(erosion, size=(ke, ke))

    img = Image.fromarray(dilation, 'L')

    contours1 = measure.find_contours(img, 0.5)

    xmed = np.zeros((2), np.uint16)
    xmaxi = np.zeros((2), np.uint16)
    xmini = np.zeros((2), np.uint16)
#     print len(contours1)
    if len(contours1) > 1:
        areaArray = []
        for i, c in enumerate(contours1):

            area = myarea(c)
            areaArray.append(area)
        sorteddata = sorted(zip(areaArray, contours1),
                            key=lambda x: x[0], reverse=True)

    # find the nth largest contour [n-1][1], in this case 2
        xmed = np.zeros(3, np.uint16)
        xmini = np.zeros(3, np.uint16)
        xmaxi = np.zeros(3, np.uint16)

        firstlargestcontour = sorteddata[0][1]
        xmin = firstlargestcontour[:, 1].min()
        xmax = firstlargestcontour[:, 1].max()
        xmed[0] = (xmax + xmin) / 2
        xmini[0] = xmin
        xmaxi[0] = xmax

        secondlargestcontour = sorteddata[1][1]
        xmin = secondlargestcontour[:, 1].min()
        xmax = secondlargestcontour[:, 1].max()
        xmed[1] = (xmax + xmin) / 2
        xmini[1] = xmin
        xmaxi[1] = xmax
        xmedf = 0
    #             print n
        ifinmin = 0
        for i in range(0, 2):
            #                 print '3', i, xmed[i],xmedf
            if xmed[i] > xmedf:

                xmedf = xmed[i]
                ifinmax = i
            else:
                ifinmin = i

        xmedian = (xmini[ifinmax] + xmaxi[ifinmin]) / 2
        if xmedian < 0.75 * dimtabx / 2 or xmedian > 1.25 * dimtabx / 2:
            xmedian = dimtabx / 2
    else:
        xmedian = dimtabx / 2
    tabMed[slicenum] = xmedian

    return tabMed


def calcSupNp(preprob, posp, lungs, imscan, pat, midx, psp, dictSubP, dimtabx):
    '''calculate the number of reticulation and HC in subpleural'''
    print 'number of subpleural for : ', pat, psp
    imgngray = np.copy(lungs)
    np.putmask(imgngray, imgngray == 1, 0)
    np.putmask(imgngray, imgngray > 0, 1)

# subErosion=  in mm
#avgPixelSpacing=0.734 in mm/ pixel
    subErosionPixel = int(round(2 * subErosion / avgPixelSpacing))
    erosion = ndimage.grey_erosion(
        imgngray, size=(
            subErosionPixel, subErosionPixel))

    np.putmask(erosion, erosion > 0, 1)
    mask_inv = np.bitwise_not(erosion)
    subpleurmask = np.bitwise_and(imgngray, mask_inv)

    ill = 0
    for ll in posp:

        xpat = ll[0]
        ypat = ll[1]

        proba = preprob[ill]
        prec, mprobai = maxproba(proba)
        classlabel = fidclass(prec, classif)

        if xpat >= midx:
            pospr = 1
            pospl = 0
        else:
            pospr = 0
            pospl = 1
        if classlabel == pat and mprobai > thrprobaUIP:
            tabpatch = np.zeros((dimtabx, dimtabx), np.uint8)
            tabpatch[ypat:ypat + dimpavy, xpat:xpat + dimpavx] = 1
            tabsubpl = np.bitwise_and(subpleurmask, tabpatch)

            area = tabsubpl.sum()
#                    check if area above threshold
            targ = float(area) / pxy

            if targ > thrpatch:
                dictSubP[pat]['all'] = (
                    dictSubP[pat]['all'][0] + pospl,
                    dictSubP[pat]['all'][1] + pospr)
                dictSubP[pat][psp] = (
                    dictSubP[pat][psp][0] + pospl,
                    dictSubP[pat][psp][1] + pospr)

        ill += 1
    return dictSubP


def maxproba(proba):
    """looks for max probability in result"""
    lenp = len(proba)
    m = 0
    for i in range(0, lenp):
        if proba[i] > m:
            m = proba[i]
            im = i
    return im, m


def fidclass(numero, classn):
    """return class from number"""
    found = False
    for cle, valeur in classn.items():

        if valeur == numero:
            found = True
            return cle

    if not found:
        return 'unknown'


def calculSurface(posp, midx, psp, dictPS):
    print 'calculate surface', psp
    ill = 0
    for ll in posp:
        xpat = ll[0]

        if xpat >= midx:
            pospr = 1
            pospl = 0
        else:
            pospr = 0
            pospl = 1

        dictPS[psp] = (dictPS[psp][0] + pospl, dictPS[psp][1] + pospr)
        dictPS['all'] = (dictPS['all'][0] + pospl, dictPS['all'][1] + pospr)

        ill += 1
    return dictPS


def diffMicro(preprob, posp, pat, midx, psp, dictP):
    '''calculate number of diffuse micronodules, left and right '''
    print 'calculate surface', pat, psp

    ill = 0
    for ll in posp:
        xpat = ll[0]

        proba = preprob[ill]
        prec, mprobai = maxproba(proba)
        classlabel = fidclass(prec, classif)
        if xpat >= midx:  # look if patch is in right or left lung
            pospr = 1
            pospl = 0
        else:
            pospr = 0
            pospl = 1

        if classlabel == pat and mprobai > thrprobaUIP:  # look if patch his above probability
            dictP[pat][psp] = (
                dictP[pat][psp][0] + pospl,
                dictP[pat][psp][1] + pospr)
            dictP[pat]['all'] = (
                dictP[pat]['all'][0] + pospl,
                dictP[pat]['all'][1] + pospr)

        ill += 1

    return dictP


def cvs(p, f, de, dse, s, dc, wf):
    '''calculate area of patches related to total area'''
    dictint = {}
    d = de[p]
    ds = dse[p]

    llungloc = (('lowerset', 'lower'), ('middleset',
                                        'middle'), ('upperset', 'upper'))
    llunglocsl = (('lowerset', 'left_sub_lower'), ('middleset',
                                                   'left_sub_middle'), ('upperset', 'left_sub_upper'))
    llunglocsr = (('lowerset', 'right_sub_lower'), ('middleset',
                                                    'right_sub_middle'), ('upperset', 'right_sub_upper'))
    llunglocl = (('lowerset', 'left_lower'), ('middleset',
                                              'left_middle'), ('upperset', 'left_upper'))
    llunglocr = (('lowerset', 'right_lower'), ('middleset',
                                               'right_middle'), ('upperset', 'right_upper'))
    if wf:
        f.write(p + ': ')
    for i in llungloc:
        st = s[i[0]][0] + s[i[0]][1]
        if st > 0:
            l = 100 * float(d[i[0]][0] + d[i[0]][1]) / st
            l = round(l, 3)
        else:
            l = 0
        dictint[i[1]] = l
        if wf:
            f.write(str(l) + ' ')

    for i in llunglocsl:
        st = s[i[0]][0]
        if st > 0:
            l = 100 * float(ds[i[0]][0]) / st
            l = round(l, 3)
        else:
            l = 0
        dictint[i[1]] = l
        if wf:
            f.write(str(l) + ' ')

    for i in llunglocsr:
        st = s[i[0]][1]
        if st > 0:
            l = 100 * float(ds[i[0]][1]) / st
            l = round(l, 3)
        else:
            l = 0
        dictint[i[1]] = l
        if wf:
            f.write(str(l) + ' ')

    for i in llunglocl:
        st = s[i[0]][0]
        if st > 0:
            l = 100 * float(d[i[0]][0]) / st
            l = round(l, 3)
        else:
            l = 0
        dictint[i[1]] = l
        if wf:
            f.write(str(l) + ' ')

    for i in llunglocr:
        st = s[i[0]][1]
        if st > 0:
            l = 100 * float(d[i[0]][1]) / st
            l = round(l, 3)
        else:
            l = 0
        dictint[i[1]] = l
        if wf:
            f.write(str(l) + ' ')

    dc[p] = dictint
    if wf:
        f.write('\n')

    return dc


def uipTree(pid, proba, posp, lungs, imscan, tabmedx,
            psp, dictP, dictSubP, dictPS, dimtabx):
    '''calculate the number of reticulation and HC in total and subpleural
    and diffuse micronodules'''
    print 'calculate volume in : ', pid

    dictPS = calculSurface(posp, tabmedx, psp, dictPS)
    print '-------------------------------------------'
    print 'surface total  by segment Left Right or:', pid
    print dictPS
    print '-------------------------------------------'
    for pat in usedclassifUIP:

        dictP = diffMicro(proba, posp, pat, tabmedx, psp, dictP)
        print pat, ' total for:', pid
        print dictP[pat]
        print '-------------------------------------------'
        dictSubP = calcSupNp(
            proba,
            posp,
            lungs,
            imscan,
            pat,
            tabmedx,
            psp,
            dictSubP,
            dimtabx)
        print pat, ' subpleural for:', pid
        print dictSubP[pat]
        print '-------------------------------------------'

    return dictP, dictSubP, dictPS


def initdictP(d, p):
    d[p] = {}
    d[p]['upperset'] = (0, 0)
    d[p]['middleset'] = (0, 0)
    d[p]['lowerset'] = (0, 0)
    d[p]['all'] = (0, 0)
    return d


def writedict(pid, pp, dx):
    (top, tail) = os.path.split(pp)
    (top, tail) = os.path.split(top)
    ftw = os.path.join(top, volumeweb)
    if not os.path.exists(ftw):
        os.mkdir(ftw)
    volumefile = open(
        os.path.join(
            ftw,
            str(dx) +
            '_' +
            str(pid) +
            '_volume.txt'),
        'w')
    volumefile.write(
        'patient UIP WEB: ' +
        str(pid) +
        ' ' +
        'patch_size: ' +
        str(dx) +
        '\n')
    volumefile.write('pattern   lower  middle  upper')
    volumefile.write('  left_sub_lower  left_sub_middle  left_sub_upper ')
    volumefile.write('  right_sub_lower  right_sub_middle  right_sub_upper ')
    volumefile.write('  left_lower  left_middle  left_upper ')
    volumefile.write(' right_lower  right_middle  right_upper\n')
    return volumefile


def manage_directories(pid):
    global errorDetected
    namedirtopf = os.path.join(flaskapp_dir, namedirtop)
    if os.path.exists(namedirtopf):
        if isinstance(pid, str):
            print 'it is a string'
            path_patient = os.path.join(namedirtopf, pid)
        else:
            path_patient = os.path.join(namedirtopf, str(pid))
        if os.path.exists(path_patient):
            lung_dir = os.path.join(path_patient, lungmask)
            lung_dir_bmp = os.path.join(lung_dir, lungmask_bmp)
            if not os.path.exists(lung_dir):
                print (lung_dir, 'does not exist full area will be used')
        else:
            print ('ERROR  path_patient:', path_patient, 'does not exist')
            errorDetected = True
    else:
        print ('ERROR top directory:', namedirtopf, 'does not exist')
        errorDetected = True

    return path_patient, lung_dir, lung_dir_bmp


def doVolume(pid):
    print "JSON Object received from the FE client patient name for volume ", str(pid)
    global errorDetected
#    dictSurf={} #dictionary with volume in each lung segment
    tabMed = {}  # dictionary with position of median between lung
    dictP = {}  # dictionary with patch in lung segment
    dictPS = {}  # dictionary with total patch area in lung segment
    dictSubP = {}  # dictionary with patch in subpleural
    dictSurf = {}  # dictionary with patch volume in percentage

    dictPS['upperset'] = (0, 0)
    dictPS['middleset'] = (0, 0)
    dictPS['lowerset'] = (0, 0)
    dictPS['all'] = (0, 0)
    for patt in classif:
        dictP = initdictP(dictP, patt)
        dictSubP = initdictP(dictSubP, patt)

    errorDetected = False
    path_patient, lung_dir, lung_dir_bmp = manage_directories(pid) # reinitialise the directory structure
    if not errorDetected:
        (top, tail) = os.path.split(path_patient)
#        print'path_patien', path_patient
        ldcm, lungSegment = definePatientSet(path_patient, lung_dir, lung_dir_bmp)  # assign  slice to lung location
        if not errorDetected:
            for scn in ldcm:
                #                print 'scannumer',scn[2]
                psp = posP(scn[2], lungSegment)
#                print scn[2],psp
                dataSlice, listLung, sizeScan = genebmp(scn[0], scn[1],path_patient)
#                cv2.imshow('listLung1',listLung)
                tabMed = calcMed(dataSlice, listLung, scn[2], tabMed, sizeScan)
                patchPositions, patchArea = generate_patches(
                    tail, dataSlice, listLung)  # generate patches for each slices
                probabilities_raw = ILDCNNpredict(patchArea)
                dictP, dictSubP, dictPS = uipTree(pid, probabilities_raw, patchPositions,
                                                  listLung, dataSlice, tabMed[scn[2]], psp, dictP, dictSubP, dictPS, sizeScan)     # generate dictionary with volume

#        print path_patient,pid
                if writeFile:
                    volumefile = writedict(pid, path_patient, dimpavx)
                else:
                    volumefile = ''
                for pat in usedclassifUIP:
                    dictSurf = cvs(
                        pat,
                        volumefile,
                        dictP,
                        dictSubP,
                        dictPS,
                        dictSurf,
                        writeFile)
                if writeFile:
                    volumefile.write('---------------------\n')
                    volumefile.close()
        else:
            print ('ERROR detected: no dicom file in :', path_patient)
    else:
        print ('ERROR detected')
    return dictSurf


def doPrediction(data, pid):
    print "JSON Object received from the FE client  ", data, "patient name ", str(pid)
    global errorDetected
    errorDetected = False
    predictionObject = {}
    # reinitialise the directory structure
    path_patient, lung_dir, lung_dir_bmp=manage_directories(pid)
    if not errorDetected:
        # assign  slice to lung location in tables
        ldcm, lungSegment = definePatientSet(path_patient, lung_dir, lung_dir_bmp)
        if data > len(ldcm):
            errorDetected = True
            print 'data: ', data, ' exceeds the number of scans in the directory:', len(ldcm)
        if not errorDetected:
            scanFound = False
            scn = ldcm[data - 1]  # pick datath slice in the set
            scanFound = True
            filedcm = scn[0]
            filelung = scn[1]
            print data, scn[2]

            if scanFound:
                dataSlice, listLung, sizeScan = genebmp(filedcm, filelung,path_patient)   # generate .bmp file out of the dicom file

                print('selected slice is : ', filedcm)

                # generate patches and return patch x/y coodinates
                patchPositions, patchArea = generate_patches(
                    filedcm, dataSlice, listLung)

                # predict process creates the probabilities of all patches over
                # entire slice rounded to 3 decimals and converted to list
                probabilities_raw = ILDCNNpredict(patchArea)
                # check if there are patches to predict or not

                if len(probabilities_raw) > 0:
                    probabilities = np.around(probabilities_raw, decimals=3)
                    probabilities_list = probabilities.tolist()

                    # generated also the averaged mean values over the entire
                    # slice
                    class_mean_values = np.around(
                        np.mean(probabilities, axis=0), decimals=3)

                    predictionObject = {'image_size': sizeScan, 'position': patchPositions, 'predicted': probabilities_list,
                                        'averaged': [
                                            {"label": "back_ground", "confidence": str(
                                                class_mean_values[classif['back_ground']])},
                                            {"label": "consolidation", "confidence": str(
                                                class_mean_values[classif['consolidation']])},
                                            {"label": "HC", "confidence": str(
                                                class_mean_values[classif['HC']])},
                                            {"label": "ground_glass", "confidence": str(
                                                class_mean_values[classif['ground_glass']])},
                                            {"label": "healthy", "confidence": str(
                                                class_mean_values[classif['healthy']])},
                                            {"label": "micronodules", "confidence": str(
                                                class_mean_values[classif['micronodules']])},
                                            {"label": "reticulation", "confidence": str(
                                                class_mean_values[classif['reticulation']])}
                                        ]}

                else:
                    print ('no patch in slice ', data, ' ', pid)
                    predictionObject = {'image_size': sizeScan, 'position': 'None', 'predicted': 'None',
                                        'averaged': [
                                            {"label": "back_ground",
                                                "confidence": 'None'},
                                            {"label": "consolidation",
                                             "confidence": 'None'},
                                            {"label": "HC", "confidence": 'None'},
                                            {"label": "ground_glass",
                                             "confidence": 'None'},
                                            {"label": "healthy",
                                             "confidence": 'None'},
                                            {"label": "micronodules",
                                             "confidence": 'None'},
                                            {"label": "reticulation",
                                             "confidence": 'None'}
                                        ]}
            else:
                print (
                    'ERROR detected: scan Number :', data, ' does not exist')
        else:
            print ('ERROR detected: no dicom file in :', path_patient)
    else:
        print ('ERROR detected')
#    visuaf.visua(
#        dataSlice,
#        patchPositions,
#        probabilities_raw,
#        dimpavx,
#        dimpavy,
#        sizeScan)
    return predictionObject

# end of function definitions
############################################

# compile the model
model = modelCompilation()
#
predictionObject=doPrediction(14,23) #takes the nth (here 10) , not the
# scan number n

#dictSurf = doVolume(23)
