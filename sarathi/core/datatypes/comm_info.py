from sarathi.utils import get_ip


class CommInfo:
    def __init__(self, driver_ip: str):
        # NOTE: In case port is already in use, this will fail.
        self.distributed_init_method = f"tcp://{driver_ip}:10000"
        self.engine_ip_address = get_ip()
        self.enqueue_socket_port = 14001
        self.output_socket_port = 14002
        self.microbatch_socket_port = 14003
        self.notify_socket_port = 14004
