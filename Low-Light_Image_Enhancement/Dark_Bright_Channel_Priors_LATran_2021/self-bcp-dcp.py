import cv2
import numpy as np
import matplotlib.pyplot as plt


def estimatedarkchannel(im,sz): # takes an input image and use a patch of sz x sz dimensions to generate the corresponding dark channel prior image
    r,g,b = cv2.split(im) # split the NumPy array representing the input RGB image into R, G, and B components respectively (each channel component becomes a NumPy array itself)
    dc = cv2.min(cv2.min(r,g),b) # perform element-wise comparison on both the NumPy array r and g to get the minimum value for each element first. Then perform element-wise comparsion on both the resulting NumPy array and NumPy array b to get the minimum value for each element. Hence, the dc is a one-channel image represented as a NumPy array whose each element is the smallest pixel value among the 3 channels (RGB) of a pixel.
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(sz,sz)) # use getStructuringElement() to manually define the kernel shape used for erode(). cv2.MORPH_RECT refers to a rectangle kernel with all its elements are the structuring elements (1's, only the image pixels overlapped by the 1's of the kernel will be considered for the operation), with the kernel size of sz x sz dimensions.
    dark = cv2.erode(dc,kernel) # as the kernel is scanned over the image dc, we compute/identify the minimal pixel value overlapped by the kernel and replace the image pixel under the anchor point (center of the kernel) with that minimal value. In short, erode() performs minimum operation [min()] on each patch of image pixels overlapped by the kernel. 
    return dark # dark, which is a one-channel image called the dark channel prior image (aka dark channel), is a NumPy array whose each element is the smallest pixel value among a patch of pixels overlapped by the kernel.


def estimatebrightchannel(im,sz): # takes an input image and use a patch of sz x sz dimensions to generate the corresponding bright channel prior image
    r,g,b = cv2.split(im) # split the NumPy array representing the input RGB image into R, G, and B components respectively (each channel component becomes a NumPy array itself)
    bc = cv2.max(cv2.max(r,g),b) # perform element-wise comparison on both the NumPy array r and g to get the maximum value for each element first. Then perform element-wise comparsion on both the resulting NumPy array and NumPy array b to get the maximum value for each element. Hence, the bc is a one-channel image represented as a NumPy array whose each element is the largest pixel value among the 3 channels (RGB) of a pixel.
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(sz,sz)) # use getStructuringElement() to manually define the kernel shape used for dilate(). cv2.MORPH_RECT refers to a rectangle kernel with all its elements are the structuring elements (1's, only the image pixels overlapped by the 1's of the kernel will be considered for the operation), with the kernel size of sz x sz dimensions.
    bright = cv2.dilate(bc,kernel) # as the kernel is scanned over the image bc, we compute/identify the maximal pixel value overlapped by the kernel and replace the image pixel under the anchor point (center of the kernel) with that maximal value. In short, dilate() performs maximum operation [max()] on each patch of image pixels overlapped by the kernel. 
    return bright # bright, which is a one-channel image called the bright channel prior image (aka bright channel), is a NumPy array whose each element is the largest pixel value among a patch of pixels overlapped by the kernel.


def guidedfilter(im,p,r,eps):
    mean_I = cv2.boxFilter(im,cv2.CV_64F,(r,r))
    mean_p = cv2.boxFilter(p, cv2.CV_64F,(r,r))
    mean_Ip = cv2.boxFilter(im*p,cv2.CV_64F,(r,r))
    cov_Ip = mean_Ip - mean_I*mean_p

    mean_II = cv2.boxFilter(im*im,cv2.CV_64F,(r,r))
    var_I   = mean_II - mean_I*mean_I

    a = cov_Ip/(var_I + eps)
    b = mean_p - a*mean_I

    mean_a = cv2.boxFilter(a,cv2.CV_64F,(r,r))
    mean_b = cv2.boxFilter(b,cv2.CV_64F,(r,r))

    q = mean_a*im + mean_b
    return q


def get_atmosphere(I, brightch, p):
    M, N = brightch.shape
    print("\n***The information within the get_atmosphere function bloack***:")
    print("The dimension of bright channel prior image [brightch], (M,N):",M,",",N)
    print("The dimension of input image [I]:", I.shape)
    flatI = I.reshape(M*N, 3) # reshape/flatten the input image I of MxN dimensions (a matrix) into M*N dimension (becomes a vector), while preserving the 3-channel pixel values of each element. In short, just change the pixel coordinates system, but not the 3-channel pixel values. So the image data still remain unchanged. The input image I is flatten from a 3D array into a 2D array so that later we can identify the RGB values of the pixels having the same indexes as the ones of the top N% largest pixel values on the bright channel prior image.
    print("The dimension of input image [I] after reshaping, flatI:", flatI.shape)
    flatbright = brightch.ravel() # make array flatten. The bright channel prior image is flatten from 2D array into 1D array so that later angsort() will consider all the pixel values on the same axis for sorting, instead of only sorting certain pixel values on a particular axis only.
    print("The dimension of bright channel prior image [brightch] after flattening, flatbright:", flatbright.shape)
    print("The input image, I:\n", I)
    print("The flatten input image, flatI:\n", flatI)
    print("The bright channel prior image, brightch:\n", brightch)
    print("The flatten bright channel prior image, flatbright:\n", flatbright)
    
    searchidx = (-flatbright).argsort()[:int(M*N*p)]  # get the top p brightest pixels from the bright channel prior image. In other words, find top M * N * p indexes. argsort() returns sorted (ascending) index.
    # -flatbright means each element of flatbright is multiplied by -1. Now, the largest pixel value (EG: 255) becomes the smallest pixel value (EG:-255).
    # since argsort() returns the sorted ascending index, now the index of the smallest pixel value will be the first element returned by argsort(). 
    # So if we want to get the indexes of the top 6 largest pixel values in flatbright, we take the first 6 elements returned by argsort(), equivalent to argsort()[:6].

    # return the mean intensity for each channel
    A = np.mean(flatI.take(searchidx, axis=0),dtype=np.float64, axis=0) # 'take' get value from index.
    # flatI.take(searchidx, axis=0) means take() takes the elements of flatI whose coordinate on axis=0 have the same number as the elements of searchidx
    # For example, let's say searchidx returns [6 5]. take(searchidx, axis=0) will take the 6th and 5th rows (because axis=0, means the first dimension of flatI) of flatI. 
    # Since flatI is a NumPy array of shape (M*N, 3), it means each row of flatI consists of 3 elements as a list. So take(searchidx, axis=0) eventually returns the 6th and 5th rows of flatI, which each of the rows is a list of 3 elements/3-channel pixel values.  
    
    # Let's say flatI.take(searchidx, axis=0) returns 2 rows whose each row is a list of 3 elements.
    # np.mean(flatI.take(searchidx, axis=0),dtype=np.float64, axis=0) means each channel of a list is added with the corresponding channel of another list, then divided by the number of rows involved (because axis=0 represents the row dimension of flatI.take(searchidx, axis=0)).
    # In other words, if N rows (each row is a list of 3 elements, representing RGB values respectively) are provided to compute the means along axis=0,
    # The final result returned by the mean() is [(summation of R channel value of all N rows/N) (summation of G channel value of all N rows/N) (summation of B channel value of all N rows/N)]=[(Averaged R channel value from the given N rows) (Averaged G channel value from the given N rows) (Averaged B channel value from the given N rows)]

    return A
    # so we first identify the first N% largest pixel values on the bright channel prior image, then get their indexes.
    # we then select the RGB channel values of the pixels on the input image that having the same indexes as the first N% largest pixel values on the bright channel prior image.
    # The selected R, G, and B channel values are added respectively and divided by the number of the selected input image pixels respectively.
    # So A = atmospheric light = [(Averaged R channel value from the number of selected pixels) (Averaged G channel value from the number of selected pixels) (Averaged B channel value from the number of selected pixels)]
    # In other words, the overall/base color of the enhanced image largely depends on the A, according to the image formation model. Because the image enhancement is performed by using A as the base value (starting value of each pixel), A and input image pixel to get the correction value, and t as the correction weight.
    # If A is a white color, then the base color of the enhanced image most probably will not experience color cast.
    # But is A is not a white color (EG:yellow color), then the base color of the enhanced image most probably will experience color cast (EG: The overall color of the enhanced image tends to be yellow color. Although the objects on the image still can be identified with their own colors respectively, their own colors will experience color cast [change of their original color] respectively)


def get_initial_transmission(A, brightch):
    A_c = np.max(A)
    
    init_t = (brightch-A_c)/(1.-A_c) # original
    
    return (init_t - np.min(init_t))/(np.max(init_t) - np.min(init_t)) # min-max normalization.


def correct_transmission(I, A, darkch, brightch, init_t, alpha, omega, w):
    im3 = np.empty(I.shape, I.dtype)

    for ind in range(0,3):
        im3[:,:,ind] = I[:,:,ind]/A[ind]

    dark_t = 1 - omega*estimatedarkchannel(im3, w)
    # dark_t = (dark_t - np.min(dark_t))/(np.max(dark_t) - np.min(dark_t))
    
    corrected_t = init_t
    diffch = brightch - darkch
    
    diff_flatten = diffch.ravel()
    indices = np.where(diff_flatten<alpha)
    
    mask = np.zeros(diff_flatten.shape)
    mask[indices] = 1
    mask_2d = mask.reshape(diffch.shape)

    inv_mask_2d = 1 - mask_2d
    
    corrected_t = dark_t*init_t*mask_2d + init_t*inv_mask_2d   

    return np.abs(corrected_t)


def get_final_image(I, A, corrected_t, tmin):
    corrected_t_broadcasted = np.broadcast_to(corrected_t[:,:,None], (corrected_t.shape[0], corrected_t.shape[1], 3))
    J = (I-A)/(np.where(corrected_t_broadcasted < tmin, tmin, corrected_t_broadcasted)) + A
    #J = (I-A)/(np.where(corrected_t < tmin, tmin, corrected_t)) + A # this is used when corrected_t has 3 channels
    # print('J between [%.4f, %.4f]' % (J.min(), J.max()))
    
    return (J - np.min(J))/(np.max(J) - np.min(J)) # min-max normalization.


def full_brightness_enhance(im, w):
    tmin=0.1   # minimum value for t to make J image
    # w=3       # window size, which determine the corseness of prior images
    alpha=0.4  # threshold for transmission correction. range is 0.0 to 1.0. The bigger number makes darker image.
    omega=0.75 # this is for dark channel prior. change this parameter to arrange dark_t's range. 0.0 to 1.0. bigger is brighter
    p=0.1      # percentage to consider for atmosphere. 0.0 to 1.0
    eps=1e-3   # for J image
    
    # Pre-process
    I = np.asarray(im, dtype=np.float64)
    I = I[:,:,:3]/255
    
    # Get dark/bright channels
    Idark_ch = estimatedarkchannel(I, w)
    Ibright_ch = estimatebrightchannel(I, w)
    
    # Get atmosphere
    # white = np.full_like(Idark, L - 1)
    At = get_atmosphere(I, Ibright_ch, p)
    
    # Get initial transmission
    init_tr = get_initial_transmission(At, Ibright_ch)
    
    # Correct transmission
    corrected_tr = correct_transmission(I, At, Idark_ch, Ibright_ch, init_tr, alpha, omega, w)
    
    # Refine transmission
    gray = cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)
    gray = np.float64(gray)/255
    refined_tr = guidedfilter(gray, corrected_tr, w, eps)
    
    # Produce final result
    enhanced_image = get_final_image(I, At, refined_tr, tmin)
    
    return enhanced_image


if __name__ == "__main__":
    
    # Load image
    img_name = "10"
    # src = f"Low-Light_Image_Enhancement/Dark_Bright_Channel_Priors_LATran_2021/images/{img_name}.jpg"
    src = f"Low-Light_Image_Enhancement/Dark_Bright_Channel_Priors_LATran_2021/images/{img_name}.bmp"
    im = cv2.imread(src) # Loads an input image from a specified file and returns the image data (pixel values) as a NumPy array of int datatype
    im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
    print("Does cv2.imread() return NumPy array?:",isinstance(im,np.ndarray))

    print("\nThe input image data (pixel values) as NumPy array (int datatype), im:\n",im)
    # plt.figure('Input image',figsize=(12,10))
    # plt.imshow(im)


	# Configuration
    tmin=0.1   # minimum value for t to make J image

	# w=15       # window size, which determine the corseness of prior images
    w=3        # window size, which determine the corseness of prior images
    alpha=0.4  # threshold for transmission correction. range is 0.0 to 1.0. The bigger number makes darker image.
    omega=0.75 # this is for dark channel prior. change this parameter to arrange dark_t's range. 0.0 to 1.0. bigger is brighter
    p=0.1      # percentage to consider for atmosphere. 0.0 to 1.0
    eps=1e-3   # for J image
    
    I = np.asarray(im, dtype=np.float64) # Convert the input image data into a NumPy array of float64 datatype.
    print("\nThe input image data (pixel values) as NumPy array (float64 datatype), I:\n",I)
    I = I[:,:,:3]/255 # Normalize the input image pixel values. stackoverflow.com/questions/44955656/how-to-convert-rgb-pil-image-to-numpy-array-with-3-channels
    print("\nThe normalized input image data (pixel values) as NumPy array (float64 datatype), I:\n",I)


	# Get Dark/Bright Channels Prior Image
    Idark_ch = estimatedarkchannel(I, w)
    Ibright_ch = estimatebrightchannel(I, w)

    # plt.figure('Dark channel prior image',figsize=(12,10))
    # plt.imshow(Idark_ch, cmap='gray') # show the dark channel prior image derived from the input image. The high pixel value on the dark channel prior image means the pixel has low contrast due to over-exposed (covered by bright region) [this property is used in this method to identify the pixel requires enhancement]. The low pixel value on the dark channel prior image means the pixel either has good contrast or has low contrast due to under-exposed (covered by dark region) or the pixel itself represents dark colored object.
    # plt.figure('Bright channel prior image',figsize=(12,10))
    # plt.imshow(Ibright_ch, cmap='gray') # show the bright channel prior image derived from the input image. The high pixel value on the bright channel prior image means the pixel either has good exposure (well-exposed) or too bright (over-exposed). The low pixel value on the bright channel prior image means the pixel either too dark (under-exposed) [this property is used in this method to identify the pixel requires enhancement] or the pixel itself represents dark colored object .

	# Get atmosphere
    At = get_atmosphere(I, Ibright_ch, p)


	# Get initial transmission and enhanced image
    init_tr = get_initial_transmission(At, Ibright_ch)


	# Correct transmission
    corrected_tr = correct_transmission(I, At, Idark_ch, Ibright_ch, init_tr, alpha, omega, w)


	# Guided filter
    gray = cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)
    gray = np.float64(gray)/255
    refined_tr = guidedfilter(gray,corrected_tr,w,eps)


	# Restore image
    refined = get_final_image(I, At, refined_tr, tmin)


	# Write out images
    output = refined*255
    output = np.array(output, dtype=np.uint8)
    cv2.imwrite(f"Low-Light_Image_Enhancement/Dark_Bright_Channel_Priors_LATran_2021/images/{img_name}_enhanced.jpg", cv2.cvtColor(output, cv2.COLOR_BGR2RGB))
    
    # plt.figure('Refined image',figsize=(12,10))
    # plt.imshow(refined)
    # plt.show()

# Insight: The resultant enhanced image contains color cast, might be due to wrong atmospheric light selection which caused by the light source (EG: lamp or moon) appears on the image [https://learnopencv.com/improving-illumination-in-night-time-images/]