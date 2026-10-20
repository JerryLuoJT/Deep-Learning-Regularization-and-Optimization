# CCDS GPU TC2 集群部署(学校虚拟机)

参考 `CCDSGPU-TC2-Masters-UserGuide.pdf`:

- Head node: `10.96.189.12`(仅限 NTU 内网,校外先连 NTU VPN
  <https://ntuvpn.ntu.edu.sg>)
- 登录:`ssh <NTU账号小写>@10.96.189.12`
- 不能直接在 head node 上跑代码,必须用 SLURM 提交作业
- 分区 `MGPU-TC2`,QoS 为分配给个人的(默认 `normal`,用
  `sacctmgr show user <username> withassoc format=user,qos` 查看)

## 步骤

1. 上传代码和数据(在 Mac 上执行):

   ```bash
   NTU_USER=<你的NTU账号> ./cluster/upload.sh
   ```

2. 登录集群并建环境(只需一次):

   ```bash
   ssh <NTU账号>@10.96.189.12
   cd ~/ai6103-homework
   bash cluster/setup_env.sh
   ```

3. 提交 Day 3 三个学习率实验:

   ```bash
   cd ~/ai6103-homework/cluster
   bash submit_all.sh
   ```

4. 查看进度和结果:

   ```bash
   squeue -u $(whoami)          # 排队/运行中的任务
   tail -f slurm_*.out          # 训练日志
   ls ~/ai6103-homework/runs/   # history.csv / 曲线图 / checkpoint
   ```

5. 下载结果回本机:

   ```bash
   rsync -avz <NTU账号>@10.96.189.12:~/ai6103-homework/runs/ ./runs/
   ```
