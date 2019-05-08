from django.shortcuts import render, redirect 
from django.http import HttpResponse

from rest_framework.views import APIView
from rest_framework import status
from rest_framework.request import Request
from rest_framework.response import Response

from rest_framework.generics import ListAPIView, RetrieveAPIView, CreateAPIView
from rest_framework.permissions import IsAdminUser

from .serializer import *
from django.conf import settings

from .forms import *

import os
from .khan_classifier.khan_classifier import get_image_class

from .cropper.cropper import get_cropped_image

class ImageCreate(CreateAPIView):
    serializer_class = ImageSerializer

    def post(self, request):
        serializer = ImageSerializer(data=request.data)
        if serializer.is_valid():
            #image_string = serializer.fields["image"].file.read()
            #img = Image.open(image_string)
            #arr = np.asarray(img)

            serializer.save()
           
            image_path = serializer.data.get('image')[1:]
            
            image_abs_path = os.path.join(settings.BASE_DIR, image_path)
            cropped_image, status = get_cropped_image(image_path)
            cropped_image.save(image_path)
            img_class = get_image_class(image_path)
            return HttpResponse('<img width=125  src="' + image_path  + '"><br>' + img_class)
        else:
            return HttpResponse('<img src="' + "not ok" + '"><br>' )






