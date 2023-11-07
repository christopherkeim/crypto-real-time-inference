// Listens for POST requests and spins up a new container
package main

import (
	"fmt"
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

		if deploy_key == os.Getenv("DOCKER_HUB_DEPLOY_KEY") {
			// Execute the bash script
			fmt.Printf("\x1b[94mStarting container upgrade\x1b[0m 🐳\n\n")
			out, err := exec.Command("deploy/deploy_container.sh").CombinedOutput()

			if err != nil {
				fmt.Printf("Error: %s", err.Error())
			}
			// Std ouput on server
			fmt.Printf("\x1b[96mThe output is:\x1b[0m\n%s\n", out)
			fmt.Printf("\x1b[36mFinished container upgrade\x1b[0m 🟢\n")
		} else {
			fmt.Printf("Invalid deployment key 🔴\n")
		}
	}).Methods("POST")

	http.ListenAndServe("127.0.0.1:8001", r)
}
