FROM python:3.7
LABEL maintainer="zuoo549674@gmail.com" owner="jason-li-831202"
COPY . /app
WORKDIR /app
RUN pip install -r requirements_docker.txt
RUN pip install --upgrade --force-reinstall "git+https://github.com/ytdl-org/youtube-dl.git"
EXPOSE 8080
CMD ["python", "Application.py"]
