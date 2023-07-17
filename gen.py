import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os

# 设定时间范围（这里设定为24小时，你可以根据需要调整）
start = datetime(2023, 1, 1)
hours = [start + timedelta(hours=i) for i in range(24*90)]

# 创建列名
prefixes = ['object_cell:', 'adjacent_cell_1:', 'adjacent_cell_2:', 'different_freq:']
fields = ['RRC_establish', 'E-RAB_establish', 'Intra_frequency_handover', 'Inter_frequency_handover', 'Intra_eNB_handover', 
          'Inter_eNB_handover', 'Uplink_PRB', 'Downlink_PRB', 'PUSCH', 'PDSCH', 'Average_user', 'Uplink_traffic', 'Downlink_traffic']
columns = ['Time', 'Delay_times_in_video_service'] + [f'{p}{f}' for p in prefixes for f in fields]

# 创建DataFrame，用随机数填充
df = pd.DataFrame(np.random.rand(len(hours), len(columns)), columns=columns)

# 设置时间列
df['Time'] = hours

# 设置时间列为index
# df.set_index('Time', inplace=True)
df.to_csv(os.path.join('data', 'cellular', 'cellular.csv'), index=False)
print('Done!')
