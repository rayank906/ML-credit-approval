## Main Architecture

Container based architecture:

1. ML model
2. Fastapi service
3. Redis Queue
4. PostgreSQL (i mean we can also use SQLite but if we wanna make this complicated we can do a dedicated container for the DB)

## Flow

1. someone POST /job with relevant data
2. FastAPI passes that information to Redis Queue, returns a job-id.
   1. Unless that job has been submitted before, we can hash the data and store that in a col and then return if it already exits
3. ML container pulls from the Redis queue, processes job, sends result back to db. maybe RAG is also done here

4. someone can use the job-id to GET /job and get the results
