import cv2
from skimage.metrics import structural_similarity as ssim


s_values = 0

for i in range(4):
    path1 = 'raihan'
    file = 'assets/' + path1 + '/' + path1 + str(i+1) + '.jpg'
    path2 = 'D:/Zayed-Work/OPUS-ML-TEAM/signature-checker/Try Own - SSIM/assets/raihan/raihan3.jpg'
    img1 = cv2.imread(file)
    img2 = cv2.imread(path2)
    # turn images to grayscale
    
    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    blur1 = cv2.GaussianBlur(gray1, (0,0), sigmaX=33, sigmaY=33)
    divide1 = cv2.divide(gray1, blur1, scale=255)
    thresh1 = cv2.threshold(divide1, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)[1]
    kernel1 = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
    morph1 = cv2.morphologyEx(thresh1, cv2.MORPH_CLOSE, kernel1)

    gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    blur2 = cv2.GaussianBlur(gray2, (0,0), sigmaX=33, sigmaY=33)
    divide2 = cv2.divide(gray2, blur2, scale=255)
    thresh2 = cv2.threshold(divide2, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)[1]
    kernel2 = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
    morph2 = cv2.morphologyEx(thresh2, cv2.MORPH_CLOSE, kernel2)

    # resize images for comparison
    img1 = cv2.resize(morph1, (300, 300))
    img2 = cv2.resize(morph2, (300, 300))
    # Check Similarity
    similarity_value = "{:.2f}".format(ssim(img1, img2)*100)
    print (i+1, " : ", similarity_value)

    s_values = s_values + float(similarity_value)

    # # display both images
    cv2.imshow("Sign Matched: " + str(similarity_value), img1)
    cv2.imshow("Input Image: ", img2)
    cv2.waitKey(0)

avg_value = s_values / 4
# Display the image
xx = "Final Match : " + str(avg_value)

print('\n' + xx + '\n')

cv2.imshow(xx, img2)
cv2.waitKey(0)
cv2.destroyAllWindows()

