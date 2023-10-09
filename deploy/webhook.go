// Listens for POST requests and spins up a new container
package main

import (
	"fmt"
	"log"
	"net/http"
	"os"
	"os/exec"

	"github.com/gorilla/mux"
)

func main() {
	// Instantiate a request router
	r := mux.NewRouter()

	// Register the request handler on the request router object
	r.HandleFunc("/infra/{deploy_key}", func(w http.ResponseWriter, r *http.Request) {
		vars := mux.Vars(r)
		deploy_key := vars["deploy_key"]

		if deploy_key == os.Getenv("DH_DEPLOY_KEY") {
			// Execute the bash script
			out, err := exec.Command("deploy/deploy_container.sh").Output()

			if err != nil {
				log.Fatal(err)
			}
			// Std ouput on server
			fmt.Printf("Starting container upgrade 🚀")
			fmt.Print(string(out))
		}
	}).Methods("POST")

	http.ListenAndServe("127.0.0.1:10000", r)
}
