{\rtf1\ansi\ansicpg936\cocoartf1504
{\fonttbl\f0\fswiss\fcharset0 Helvetica;\f1\fnil\fcharset0 Menlo-Regular;}
{\colortbl;\red255\green255\blue255;\red0\green0\blue0;\red28\green0\blue207;}
{\*\expandedcolortbl;\csgray\c100000;\csgenericrgb\c0\c0\c0;\csgenericrgb\c11000\c0\c81000;}
\paperw11900\paperh16840\margl1440\margr1440\vieww10800\viewh8400\viewkind0
\pard\tx566\tx1133\tx1700\tx2267\tx2834\tx3401\tx3968\tx4535\tx5102\tx5669\tx6236\tx6803\pardirnatural\partightenfactor0

\f0\fs24 \cf0 Hybrid.py\
\
This code is designed for generating hybrid image. \
\
1. You can modify the input of code to get different image, as below\
\
\pard\tx543\pardeftab543\pardirnatural\partightenfactor0

\f1\fs22 \cf2 \CocoaLigature0 	image1 = cv2.imread(\cf3 '../data/einstein.bmp'\cf2 ) / \cf3 255.0\cf2 \
    image2 = cv2.imread(\cf3 '../data/marilyn.bmp'\cf2 ) / \cf3 255.0\
\

\f0\fs24 \cf0 	By adding an extra parameter into \'93
\f1\fs22 \cf2 imread
\f0\fs24 \cf0 \'94, we could get gray-scale image, which is also 		applicable in this code.\
\
2. You can use different function to use different algorithm to generate image\

\f1\fs22 \cf2 \
    low_frequencies = im_imfilter(image1, kernel)\
    high_frequencies = image2 - im_imfilter(image2, kernel)\
\

\f0\fs24 \cf0 	We utilize convolution method, that is, im_filter function by default. If you want to use 			Fast Fourier Transformation to generate image, you can replace 
\f1\fs22 \cf2 im_imfilter
\f0\fs24 \cf0  with \
	
\f1\fs22 \cf2 fft_imfilter.\
\

\f0\fs24 \cf0 3. You can freely set different cutoff frequency.\
	\

\f1\fs22 \cf2 	cutoff_frequency = \cf3 5\
\
	
\f0\fs24 \cf0 Also, we provide a default cutoff_frequncy, if you want to see the result in different cutoff 		frequency, feel free to modify it.\
\
Hint: It is not recommended to modify other part of code.}