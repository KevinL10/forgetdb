use anyhow::Result;
use chrono::{DateTime, Utc};
use qdrant_client::prelude::*;
use qdrant_client::qdrant::qdrant_server::Qdrant;
use qdrant_client::qdrant::value::Kind;
use qdrant_client::qdrant::vectors::VectorsOptions;
use qdrant_client::qdrant::vectors_config::Config;
use qdrant_client::qdrant::{
    CollectionOperationResponse, Condition, CreateCollection, Filter, PointsSelector, SearchPoints,
    SearchResponse, Vector, VectorParams, VectorsConfig,
};
use rand::Rng;
use serde_json::json;
use std::borrow::Borrow;
use std::collections::BinaryHeap;
use std::future::IntoFuture;
use std::sync::Arc;
use std::time::Duration;
use tokio::sync::Mutex;
use tokio::task::JoinHandle;
use tokio::time;
use uuid::Uuid;

struct RemembranceClient {
    // half life time of noise in seconds
    time_decay: f64,
    embedding_len: usize,
    qdrant_client: QdrantClient,
    // A heap storing (-lastProcessedTime, creationTime, id)
    process_queue: BinaryHeap<(i64, i64, String)>,
}

impl RemembranceClient {
    pub fn new(time_decay: f64, embedding_len: usize, url: &str) -> Result<Self> {
        let client = QdrantClient::from_url(url)
            .with_api_key(std::env::var("QDRANT_API_KEY"))
            .build()?;

        Ok(Self {
            time_decay,
            embedding_len,
            process_queue: BinaryHeap::new(),
            qdrant_client: client,
        })
    }

    pub fn init_forget(arc_self: Arc<Mutex<Self>>) -> JoinHandle<i32> {
        tokio::spawn(async move {
            loop {
                let mut self_lock = arc_self.lock().await;

                if let Some(head) = self_lock.process_queue.pop() {
                    // let mut self_lock = arc_self.lock().await;
                    let (last_updated_time, creation_time, id) = head;
                    // Assuming `get_points` is an async function
                    let points = self_lock
                        .qdrant_client
                        .get_points(
                            "test",
                            None,
                            &[id.to_string().into()], // Assuming id is already a String
                            Some(true),
                            Some(true),
                            None,
                        )
                        .await
                        .expect("get points failed");

                    // println!("updated point {:?} {:?}", id, lastUpdatedTime);
                    let res = self_lock
                        .qdrant_client
                        .delete_points("test", None, &vec![id.to_string().into()].into(), None)
                        .await;
                    // println!("{:?}", res);
                    let vectors = points.result[0]
                        .vectors
                        .as_ref()
                        .unwrap()
                        .vectors_options
                        .as_ref()
                        .unwrap();

                    let mut embedding = match vectors {
                        VectorsOptions::Vector(s) => s.data.clone(),
                        _ => panic!("no named vectors"),
                    };
                    let time_in_seconds = Utc::now().timestamp();

                    for value in embedding.iter_mut() {
                        let random_number = (rand::random::<f64>() - 0.5) * 2.0 * (time_in_seconds as f64 + last_updated_time as f64) / 15.0 ; // Generate a random number between -1.0 and 1.0
                        let random_number = (rand::random::<f64>() - 0.5) * 2.0;
                        *value += random_number as f32; // Add the random number to the original value
                    }

                    let payload = points.result[0]
                        .payload
                        .get("data")
                        .unwrap()
                        .kind
                        .as_ref()
                        .unwrap();

                    // println!("reinserting point {:?}", id);
                    self_lock
                        .process_queue
                        .push((-time_in_seconds, creation_time, id.clone()));

                    let point = PointStruct::new(
                        id.to_string(),
                        embedding,
                        json!({"data": match payload {
                            Kind::StringValue(s) => &s,
                            _ => "",
                        }})
                        // json!({"data": "test"})
                        .try_into()
                        .unwrap(),
                    );
                    let _ = self_lock
                        .qdrant_client
                        .upsert_points_blocking("test", None, vec![point], None)
                        .await;
                }
            }
        })
    }

    pub async fn delete_collection(&self, collection_name: &str) -> Result<()> {
        self.qdrant_client
            .delete_collection(collection_name)
            .await?;
        Ok(())
    }

    pub async fn create_collection(&self, collection_name: &str) -> Result<()> {
        self.qdrant_client
            .create_collection(&CreateCollection {
                collection_name: collection_name.into(),
                vectors_config: Some(VectorsConfig {
                    config: Some(Config::Params(VectorParams {
                        size: self.embedding_len as u64,
                        distance: Distance::Cosine.into(),
                        ..Default::default()
                    })),
                }),
                ..Default::default()
            })
            .await?;
        Ok(())
    }

    pub async fn add(
        &mut self,
        collection_name: &str,
        embedding: Vec<f32>,
        payload: &str,
        significance: f64,
    ) -> Result<()> {
        let id = Uuid::new_v4().to_string();
        let time_in_seconds = Utc::now().timestamp();
        self.process_queue
            .push((-time_in_seconds, time_in_seconds, id.clone()));

        let point = PointStruct::new(id, embedding, json!({"data": payload}).try_into().unwrap());
        self.qdrant_client
            .upsert_points_blocking(collection_name, None, vec![point], None)
            .await?;

        Ok(())
    }

    pub async fn query(
        &self,
        collection_name: &str,
        embedding: Vec<f32>,
        n: u64,
    ) -> Result<SearchResponse> {
        self.qdrant_client
            .search_points(&SearchPoints {
                collection_name: collection_name.into(),
                vector: embedding,
                // filter: Some(Filter::all([Condition::matches("bar", 12)])),
                limit: n,
                with_payload: Some(true.into()),
                ..Default::default()
            })
            .await
    }
}

fn print_response(response: SearchResponse) {
    for scored_point in response.result.into_iter() {
        let payload = scored_point.payload
            .get("data")
            .unwrap()
            .kind
            .as_ref()
            .unwrap();
        let score = scored_point.score;
        println!("{:?}: {:?}", payload, score);
    }
    println!("");
}

#[tokio::main]
async fn main() -> Result<()> {
    let length = 2;
    let client = RemembranceClient::new(
        1.0,
        length,
        "https://b7e4ef34-14b5-40f5-a37c-859f8fb76b09.us-east4-0.gcp.cloud.qdrant.io:6334",
    )?;

    let client = Arc::new(Mutex::new(client));

    // let client = Arc::new(Mutex::new(client));
    // RemembranceClient::init_forget(client.clone());
    let handle = RemembranceClient::init_forget(client.clone());

    {
        let mut client_lock = client.lock().await;
        client_lock.delete_collection("test").await?;
        client_lock.create_collection("test").await?;

        client_lock
            .add("test", vec![0.2, 0.3], "a yellow banana", 1.0)
            .await?;
        client_lock
            .add("test", vec![0.4, 0.4], "a blue banana", 1.0)
            .await?;
        client_lock
            .add("test", vec![-0.2, 0.8], "a red banana", 1.0)
            .await?;

        let results = client_lock.query("test", vec![0.4, 0.58], 3).await?;
        print_response(results);
    }

    loop {
        time::sleep(Duration::from_secs(5)).await;
        let mut client_lock = client.lock().await;
        let results = client_lock.query("test", vec![0.4, 0.58], 3).await?;
        print_response(results);
    }
}
