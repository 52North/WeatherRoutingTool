FROM python:3.13

#
# Update system dependencies
#
RUN apt-get update \
 && apt-get upgrade -y \
 && apt-get dist-upgrade -y \
 && apt-get clean \
 && apt autoremove -y  \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY ./WeatherRoutingTool ./WeatherRoutingTool

COPY ./pyproject.toml ./requirements.txt ./requirements-without-deps.txt ./cli.py ./

RUN pip install . && pip install --no-deps -r requirements-without-deps.txt

ENTRYPOINT [ "python", "/app/cli.py", \
             "--file", "/app/wrt_work_dir/config.json", \
             "--info-log-file", "/app/wrt_work_dir/logs/info.log" ]