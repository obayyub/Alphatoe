#!/bin/bash
LOG_DIR="training_logs"
mkdir -p $LOG_DIR
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

# Log kernel messages
sudo dmesg -w > "$LOG_DIR/dmesg_$TIMESTAMP.log" &

# Log system messages  
sudo journalctl -f > "$LOG_DIR/journal_$TIMESTAMP.log" &

# Log NVIDIA errors specifically
nvidia-smi -l 5 > "$LOG_DIR/nvidia_smi_$TIMESTAMP.log" &

# Log memory statistics
vmstat 5 > "$LOG_DIR/vmstat_$TIMESTAMP.log" &

echo "Logging to $LOG_DIR/"
echo "Timestamp: $TIMESTAMP"
echo "Press Ctrl+C to stop all logging"
wait
