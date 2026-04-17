ros2 run paper_bag_detection real_to_bag
 
---------------------------------
colcon build --symlink-install --packages-select paper_bag_detection (<---- in my_ws)

If you see an error like the one below, please rebuild the package. 

ex)

Exception in thread Thread-1 (inference_thread):

Traceback (most recent call last):

inference_sdk.http.errors.APIKeyNotProvided: API key must be provided in this case

