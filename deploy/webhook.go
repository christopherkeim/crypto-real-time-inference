// Listens for POST requests and rolls an updated container
package main

import (
	"bytes"
	"encoding/json"
	"fmt"
	"net/http"
	"os"
	"os/exec"

	"github.com/gorilla/mux"
)

type Webhook struct {
	CallbackURL string `json:"callback_url"`
}

type ValidationResponse struct {
	State string `json:"state"`
}

func main() {
	// Instantiate a request router
	router := mux.NewRouter()

	// Register the request handler on the request router
	router.HandleFunc("/infra/{deploy_key}", func(writer http.ResponseWriter, request *http.Request) {
		vars := mux.Vars(request)
		deployKey := vars["deploy_key"]

		if deployKey == os.Getenv("DOCKER_HUB_DEPLOY_KEY") {
			// Parse the webhook body and extract the callback url
			request.Body = http.MaxBytesReader(writer, request.Body, 1048576)
			decoder := json.NewDecoder((request.Body))
			var webhook Webhook
			parsingError := decoder.Decode(&webhook)

			if parsingError != nil {
				fmt.Printf("Error: %s", parsingError.Error())
			}

			// Construct the validation response
			validationResponse := ValidationResponse{State: "success"}
			jsonResponse, jsonError := json.Marshal(validationResponse)

			if jsonError != nil {
				fmt.Printf("Error: %s", jsonError.Error())
			}

			// Validate the webhook by posting to the callback url
			_, responseError := http.NewRequest("POST", webhook.CallbackURL, bytes.NewBuffer(jsonResponse))
			if responseError != nil {
				fmt.Printf("Error: %s", responseError.Error())
			}

			fmt.Printf("\x1b[96mSuccessfully validated webhook\x1b[0m ✅\n")

			// Execute the deployment
			fmt.Printf("\x1b[96mStarting container upgrade\x1b[0m 🐳\n\n")
			output, deployError := exec.Command("deploy/deploy_container.sh").CombinedOutput()

			if deployError != nil {
				fmt.Printf("Error: %s", deployError.Error())
			}

			fmt.Printf("\x1b[96mThe output is:\x1b[0m\n%s\n", output)
			fmt.Printf("\x1b[96mFinished container upgrade\x1b[0m 🟢\n")
		} else {
			fmt.Printf("\x1b[96mInvalid deployment key\x1b[0m 🔴\n")
		}
	}).Methods("POST")

	http.ListenAndServe("127.0.0.1:8001", router)
}
