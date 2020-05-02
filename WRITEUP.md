# Project Write-Up

## Model Usage and Application Logic

* Converting model ([MobileNetSSD](https://www.pyimagesearch.com/2017/09/11/object-detection-with-deep-learning-and-opencv/))

The downloaded as well as optimized model can be found in /models
```console
cd /opt/intel/openvino/deployment_tools/model_optimizers/

sudo ./mo.py --input_model <path to MobileNetSSD_deploy.caffemodel>
--input_proto <path to MobileNetSSD_deploy.prototxt> -o <output folder>
--mean_values [127.5,127.5,127.5] --scale_values [127.5]
```

* Counting logic

Counting people is done by detecting enter and leave action. The naive solution relies on the fact that there is always max 1 person on frame and the person is always entering from the bottom and leaving from the right. Enter action is detected when the personInFrame is bigger than lastCount and there is overlap between the bounding box and 5%-bottom-image. Leave action is analog detected when the personInFrame is smaller than lastCount and there is overlap between the bounding box and 5%-right-image

## Explaining Custom Layers

There is no custom layers in my pre-trained model.

However, handling custom layers is in general a very important step. Openvino tool kit by default only supports a list of layers, which also called "supported layers". There are also supported layers specific by hardware. The layers outside of this list is considered as "custom layers". By default, the inference engine will report an error while loading trained model containing those "custom layers". Handling those custom layers will help using your model successfully and optimizedly.

## Comparing Model Performance

I used test.jpg in resources/test.jpg to test the optimized model by openvino and the model itself (using cv2.dnn library). The results are saved in resources/ original_model.png and optimized_model.jpg

* the accuracy by optimized model is almost 10% better. Concretely in the test image there are 2 people detected with the accuracy as following (see in image results)

|model | origin | optimized |
| --- | --- | --- |
|person 1 | 86.79%| **94.89%**|
|person 2 | 26.94%| **32.05%** |

* the size of the optimized is about 10kB smaller than the original model. Plus in the converting step as mentioned above, a preprocessing step (normalization) is put directly into the optimized model per --mean_values and --scale_values so that there is no need to normalize the image before feeding the model anymore.

```console
origin
-rw-r--r--  1 tali tali 23147564 Aug 27  2017 MobileNetSSD_deploy.caffemodel
-rw-r--r--  1 tali tali    29353 Aug 27  2017 MobileNetSSD_deploy.prototxt

optimized
-rw-r--r-- 1 root root 23133788 Apr 25 14:27 MobileNetSSD_deploy.bin
-rw-r--r-- 1 root root   177101 Apr 25 14:27 MobileNetSSD_deploy.xml
```

* The inference time of the optimized is shorter than the original model. Concretely the time to inference the test.jpg is 0.128113985062 and **0.11425471305847168** (optimized)

## Assess Model Use Cases

Some of the potential use cases of the people counter app are:

* One general usecase is for room/ship/train/cinemax/disco/shop/place with limited number of people. A people counter app is the best help. With it, the resource capacity is guaranteed for usage and corresponding to it, a high level customer service quality is also met.
* A people counter app is especially helpful in the current corona crise, where it is necessary to limit the number of people entering critical places such as restaurants, supermarks, bars. As we experienced in Germany, it is currently manually solved by one employee, who controls the number of people entering such places. A people counter app will automatic this process and therefore save the human effort, cost and money.
* Another potential use case I see in the future is completely automatic shopping. It means there is no staff needed. An example is amazon shop. For running such a shop, it is very important to have a system, which calculates the people entering to guarantee that there are not too many people shopping in a time, which contributes to the customer satisfaction.

## Assess Effects on End User Needs

* An end user is in this case in my opinion the one who runs/admins the application.
* A consideration about the trade-off between accuracy and cost must be done carefully. A good model sometimes has bigger architecture, requires more computation and requires more effort (money + human) to build
* Camera focal length/image size depends a lot on the used model and use cases. How good can the model deal with such distance or variable image size? Is a lost in accuracy caused by such reasons acceptable? Is it tolerable in the use cases?
