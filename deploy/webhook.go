// Listens for POST requests and rolls out an updated container
package main

import (
	"bytes"
	"encoding/json"
	"fmt"
	"net/http"
	"os"
	"os/exec"

	"github.com/go-chi/chi/v5"
)

type Webhook struct {
	CallbackURL string `json:"callback_url"`
}

type ValidationResponse struct {
	State string `json:"state"`
}

func main() {
	// Instantiate a request router
	router := chi.NewRouter()

	// Register the request handler on the request router
	router.Post("/infra/{deploy_key}", func(writer http.ResponseWriter, request *http.Request) {
		deployKey := chi.URLParam(request, "deploy_key")

		if deployKey != os.Getenv("DOCKER_HUB_DEPLOY_KEY") {
			fmt.Printf("\x1b[96mInvalid deployment key\x1b[0m 🔴\n")
			return
		}

		// Parse the webhook body and extract the callback url
		request.Body = http.MaxBytesReader(writer, request.Body, 1048576)
		decoder := json.NewDecoder((request.Body))
		var webhook Webhook
		parsingError := decoder.Decode(&webhook)

		if parsingError != nil {
			fmt.Printf("Parsing Error: %s", parsingError.Error())
		}

		// Construct the validation response
		validationResponse := ValidationResponse{State: "success"}
		jsonResponse, jsonError := json.Marshal(validationResponse)

		if jsonError != nil {
			fmt.Printf("JSON Error: %s", jsonError.Error())
		}

		// Validate the webhook by posting to the callback url
		_, responseError := http.NewRequest("POST", webhook.CallbackURL, bytes.NewBuffer(jsonResponse))
		if responseError != nil {
			fmt.Printf("Validation Response Error: %s", responseError.Error())
		}

		fmt.Printf("\x1b[96mSuccessfully validated webhook\x1b[0m ✅\n")

		// Execute the deployment
		fmt.Printf("\x1b[96mStarting container upgrade\x1b[0m 🐳\n\n")
		output, deployError := exec.Command("deploy/deploy_container.sh").CombinedOutput()

		if deployError != nil {
			fmt.Printf("Deployment Error: %s", deployError.Error())
		}

		fmt.Printf("\x1b[96mThe output is:\x1b[0m\n%s\n", output)
		fmt.Printf("\x1b[96mFinished container upgrade\x1b[0m 🟢\n")

	})

	http.ListenAndServe("127.0.0.1:8001", router)
}
