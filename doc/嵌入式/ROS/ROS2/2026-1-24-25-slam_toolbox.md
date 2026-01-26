# slam_toolbox

这是一个二维的, 开箱即用的一个建模工具

```bash
sudo apt-get install ros-$ROS_DISTRO-slam-toolbox
ros2 launch slam_toolbox online_async_launch.py use_sim_time:=True
```

![image-20260124142219219](https://picture-01-1316374204.cos.ap-beijing.myqcloud.com/lenovo-picture/202601241422388.png)

![image-20260124143450100](https://picture-01-1316374204.cos.ap-beijing.myqcloud.com/lenovo-picture/202601241434298.png)

![image-20260124143609516](https://picture-01-1316374204.cos.ap-beijing.myqcloud.com/lenovo-picture/202601241436630.png)

## 地图保存

```bash
sudo apt-get install ros-$ROS_DISTRO-nav2-map-server
ros2 run nav2_map_server map_saver_cli -f room
```

![image-20260124145705565](https://picture-01-1316374204.cos.ap-beijing.myqcloud.com/lenovo-picture/202601241457650.png)
