from sarathi.utils import get_ip, get_random_port


class CommInfo:
    def __init__(self, driver_ip: str):
        # NOTE: In case port is already in use, this will fail.
        self.distributed_init_method = f"tcp://{driver_ip}:10000"
        self.engine_ip_address = get_ip()
        self.enqueue_socket_port = 10001
        self.output_socket_port = 10002
        self.microbatch_socket_port = 10003
