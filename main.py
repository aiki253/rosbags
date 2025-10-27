from rosbags.rosbag2 import Reader
from rosbags.typesys import Stores, get_typestore
import numpy as np

# typestoreの作成
typestore = get_typestore(Stores.LATEST)

with Reader('/Users/aiki/PythonCode/rosbags/rosbag2_2025_10_08-15_15_28') as reader:
    print("=== ROSバッグのデータ構造 ===\n")
    
    # 全トピックの情報を表示
    print("利用可能なトピック:")
    for connection in reader.connections:
        print(f"  - トピック: {connection.topic}")
        print(f"    メッセージタイプ: {connection.msgtype}")
        print(f"    メッセージカウント: {connection.msgcount}")
        print()
    
    # /scanトピックの詳細
    print("\n=== /scan トピックの詳細 ===")
    scan_connections = [x for x in reader.connections if x.topic == '/scan']
    if scan_connections:
        connection, timestamp, rawdata = next(reader.messages(connections=scan_connections))
        msg = typestore.deserialize_cdr(rawdata, connection.msgtype)
        
        print(f"フィールド: {dir(msg)}")
        print(f"\nヘッダー:")
        print(f"  - frame_id: {msg.header.frame_id}")
        print(f"  - timestamp: {msg.header.stamp.sec}.{msg.header.stamp.nanosec}")
        print(f"\nスキャン設定:")
        print(f"  - 角度範囲: {np.degrees(msg.angle_min):.1f}° ~ {np.degrees(msg.angle_max):.1f}°")
        print(f"  - 角度増分: {np.degrees(msg.angle_increment):.3f}°")
        print(f"  - 距離範囲: {msg.range_min}m ~ {msg.range_max}m")
        print(f"  - スキャン時間: {msg.scan_time}秒")
        print(f"  - time_increment: {msg.time_increment}秒")
        print(f"  - データポイント数: {len(msg.ranges)}")
        print(f"  - 強度データ数: {len(msg.intensities)}")
    
    # /joyトピックの詳細
    print("\n=== /joy トピックの詳細 ===")
    joy_connections = [x for x in reader.connections if x.topic == '/joy']
    if joy_connections:
        connection, timestamp, rawdata = next(reader.messages(connections=joy_connections))
        msg = typestore.deserialize_cdr(rawdata, connection.msgtype)
        
        print(f"フィールド: {dir(msg)}")
        print(f"\nヘッダー:")
        print(f"  - frame_id: {msg.header.frame_id}")
        print(f"  - timestamp: {msg.header.stamp.sec}.{msg.header.stamp.nanosec}")
        print(f"\nジョイスティックデータ:")
        print(f"  - axes (軸の数): {len(msg.axes)}")
        print(f"  - axes値: {msg.axes}")
        print(f"  - buttons (ボタンの数): {len(msg.buttons)}")
        print(f"  - buttons値: {msg.buttons}")
        
        # 最初の10個のjoyメッセージを表示
        print(f"\n最初の10個のjoyメッセージ:")
        reader_reset = Reader('/Users/aiki/PythonCode/rosbags/rosbag2_2025_10_08-15_15_28')
        reader_reset.__enter__()
        joy_connections = [x for x in reader_reset.connections if x.topic == '/joy']
        for i, (connection, timestamp, rawdata) in enumerate(reader_reset.messages(connections=joy_connections)):
            if i >= 10:
                break
            msg = typestore.deserialize_cdr(rawdata, connection.msgtype)
            print(f"  {i+1}. axes: {msg.axes}, buttons: {msg.buttons}")
        reader_reset.__exit__(None, None, None)
    else:
        print("  /joyトピックが見つかりません")
    
    # 両方のトピックを同時に処理する例
    print("\n=== /scanと/joyを同時に取得 ===")
    target_connections = [x for x in reader.connections if x.topic in ['/scan', '/joy']]
    
    scan_count = 0
    joy_count = 0
    
    for connection, timestamp, rawdata in reader.messages(connections=target_connections):
        msg = typestore.deserialize_cdr(rawdata, connection.msgtype)
        
        if connection.topic == '/scan':
            scan_count += 1
        elif connection.topic == '/joy':
            joy_count += 1
            if joy_count <= 5:  # 最初の5個だけ表示
                print(f"Joy {joy_count}: axes={msg.axes}, buttons={msg.buttons}")
    
    print(f"\n合計 - /scan: {scan_count}個, /joy: {joy_count}個")