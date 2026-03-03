import sys
import os
import threading
import time

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.monitor import NetworkHealthMonitor
from src.sender import TransactionProber

def main():
    print(" Starting PROFESSIONAL Data Collection Pipeline...")
    print("   -> Mode: Parallel Execution (Monitor + Sender)")
    
    # 1. Start the Monitor (Background Thread)
    # It will ping RPCs every 30 seconds silently
    monitor = NetworkHealthMonitor(interval=30)
    monitor_thread = threading.Thread(target=monitor.start)
    monitor_thread.daemon = True # This ensures it dies when main program dies
    monitor_thread.start()
    print("✅ Network Monitor started in background.")

    # 2. Start the Sender (Main Thread)
    # We will collect 350 MORE transactions to hit the 1000 mark
    target_tx = 350 
    sender = TransactionProber(max_tx=target_tx)
    
    try:
        sender.run_batch()
    except KeyboardInterrupt:
        print("\n Stopping collection...")
    
    # 3. Cleanup
    monitor.stop()
    print(" Data Collection Complete.")

if __name__ == "__main__":
    main()