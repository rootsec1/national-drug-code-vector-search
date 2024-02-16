# Use an official Python runtime as a parent image
FROM python:3.10

# Set the working directory in the container to /app
WORKDIR /app

# Download and extract the SQLite source code
RUN wget https://www.sqlite.org/2022/sqlite-autoconf-3380500.tar.gz && \
    tar xvfz sqlite-autoconf-3380500.tar.gz && \
    cd sqlite-autoconf-3380500

RUN pwd && ls -l && chmod +x configure && ./configure --disable-tcl --enable-shared --enable-tempstore=always --prefix="$PREFIX"

# Configure, build, and install SQLite
RUN export CFLAGS="-DSQLITE_ENABLE_FTS3 \
-DSQLITE_ENABLE_FTS3_PARENTHESIS \
-DSQLITE_ENABLE_FTS4 \
-DSQLITE_ENABLE_FTS5 \
-DSQLITE_ENABLE_JSON1 \
-DSQLITE_ENABLE_LOAD_EXTENSION \
-DSQLITE_ENABLE_RTREE \
-DSQLITE_ENABLE_STAT4 \
-DSQLITE_ENABLE_UPDATE_DELETE_LIMIT \
-DSQLITE_SOUNDEX \
-DSQLITE_TEMP_STORE=3 \
-DSQLITE_USE_URI \
-O2 \
-fPIC" && \
    export PREFIX="/usr/local" && \
    LIBS="-lm" ./configure --disable-tcl --enable-shared --enable-tempstore=always --prefix="$PREFIX" && \
    make && \
    make install

# Add the current directory contents into the container at /app
ADD . /app

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Make port 80 available to the world outside this container
EXPOSE 80

# Run app.py when the container launches
CMD ["python", "app.py"]
